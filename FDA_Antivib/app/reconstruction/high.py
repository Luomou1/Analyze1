"""EXE High 0xABC060 的高度细化与残差修正。

采用用户实际N帧/物理扫描坐标适配；不是DLL FringeOrderAnalysis模式2。
"""
import numpy as np
from app.reconstruction.spectrum import SpectrumResult
from app.reconstruction.high_regions import reject_discontinuities,unwrap_regions
from app.reconstruction.high_residual import subtract_plane,subtract_histogram_mode,wrap_high_residual


def histogram_phase_reference(alpha):
    """0xAA8EE0：周期化、1000桶、15桶循环求和，最大窗口中点取偏差参考。"""
    values=np.fmod(alpha[np.isfinite(alpha)].astype(np.float32).astype(float),2*np.pi).astype(np.float32)
    values=np.where(values<0,values+np.float32(2*np.pi),values)
    if not values.size:raise ValueError('High无有效相位。')
    scaled=np.trunc(values.astype(np.float32)*np.float32(1000)).astype(np.int64)
    lo=int(scaled.min());hi=int(scaled.max())
    if hi==lo:return lo/1000
    # 0xA7E070按整数值覆盖宽度分配到相邻两桶；并非简单计数直方图。
    width=(hi-lo+1)/1000
    position=(scaled-lo)/width
    index=np.trunc(position).astype(np.int64)
    split=position-index>(width-1)/width
    first=np.where(split,np.trunc(((index+1)*width-(scaled-lo))*1000+.5),1000).astype(np.int64)
    hist=np.bincount(index,weights=first,minlength=1000).astype(np.int64)
    second=split&(index+1<1000)
    hist+=np.bincount(index[second]+1,weights=1000-first[second],minlength=1000).astype(np.int64)
    hist=(hist+500)//1000
    sums=sum(np.roll(hist,-i) for i in range(15))
    center=(int(np.argmax(sums))+7)%1000
    return float((np.float32(center)/1000*np.float32(hi-lo)+lo)/1000)


def reconstruct_high(front: SpectrumResult):
    tau=2*np.pi
    alpha=-front.intercept
    reference=histogram_phase_reference(alpha)
    rounds=np.trunc((alpha-reference+np.where(alpha-reference>0,np.pi,-np.pi))/tau)
    physical_k=tau*front.peak_bin/(front.frame_count*front.step_um)
    fine=front.reference_um+(front.phase+rounds*tau)/physical_k
    # EXE残差先转载频归一化整数域；一条纹=4096，不以纳米直接拟合。
    scale=physical_k*4096/tau
    residual=np.trunc((fine-front.coarse_um)*scale)
    filtered=reject_discontinuities(residual)
    connected,labels,counts=unwrap_regions(filtered)
    output=fine.copy()
    accepted=counts.sum() if counts.size else 0
    # 0xABC97C/0xABC986/0xABC9B0：10%、20点、主域30%的显式条件。
    if accepted>front.valid.sum()*.1 and accepted>20 and counts.max()>accepted*.3:
        mask=labels==int(np.argmax(counts))
        y,x=np.indices(mask.shape)
        design=np.column_stack((x[mask],y[mask],np.ones(mask.sum())))
        if np.linalg.matrix_rank(design)<3:
            raise ArithmeticError('High主区域平面拟合奇异。')
        plane=np.linalg.solve(design.T@design,design.T@connected[mask])
        detrended=subtract_plane(np.where(mask,connected,np.nan),plane)
        centered,_=subtract_histogram_mode(detrended)
        corrected=wrap_high_residual(centered,4096)
        output=front.coarse_um+corrected/scale
    return output,{'high_phase_reference_rad':reference,'high_residual':residual,
                   'high_connected':connected,'high_regions':labels}
