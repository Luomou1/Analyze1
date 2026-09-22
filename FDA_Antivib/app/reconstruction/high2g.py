"""EXE High 2G 单材料、无色散标定分支；不调用DLL FDA级次处理。"""
import numpy as np
from scipy.ndimage import convolve
from app.reconstruction.spectrum import SpectrumResult
from app.reconstruction.integer_regions import connect_regions

PERIOD=4096


def align_period(values,reference,period):
    delta=values-reference
    turns=np.where(delta>period/2,np.ceil((delta-period/2)/period),
                   np.where(delta < -period/2,np.floor((delta+period/2)/period),0))
    return values-period*turns


def reconstruct_high2g(front: SpectrumResult,common_k0: float | None=None):
    valid=front.valid
    if not valid.any():raise ValueError('High 2G 无有效频谱像素。')
    # 界面K0是光学波数，内部强度随位移的角频率为2*K0。
    if common_k0 is not None and (not np.isfinite(common_k0) or common_k0<=0):
        raise ValueError("手动公共K0必须为有限正数。")
    carrier=(float(front.peak_bin[valid].mean()) if common_k0 is None
             else common_k0*front.frame_count*front.step_um/np.pi)
    k=2*np.pi*carrier/(front.frame_count*front.step_um)
    scale=PERIOD*k/(2*np.pi)
    phase=front.intercept+front.slope*carrier
    fine_um=front.reference_um+phase/k
    coarse=np.where(valid,front.coarse_um*scale,0)
    coarse=np.trunc(coarse+np.copysign(.5,coarse)).astype(np.int64)
    fine=np.where(valid,fine_um*scale,0)
    fine=np.trunc(fine+np.copysign(.5,fine)).astype(np.int64)
    kernel=np.ones((3,3),dtype=np.int64)
    counts=convolve(valid.astype(np.int64),kernel,mode='constant')
    sums=convolve(np.where(valid,coarse,0),kernel,mode='constant')
    mean=coarse.astype(float);np.divide(sums,counts,out=mean,where=counts>0)
    filtered=coarse+np.clip(np.trunc(mean).astype(np.int64)-coarse,-PERIOD//4,PERIOD//4)
    raw_gap=fine-filtered
    angles=raw_gap[valid]*2*np.pi/PERIOD
    average=float(np.arctan2(np.sin(angles).sum(),np.cos(angles).sum())*PERIOD/(2*np.pi))
    fine=align_period(fine,filtered+average,PERIOD).astype(np.int64)
    connection=connect_regions(fine-filtered,valid,period=PERIOD,noise_percent=10)
    observed=connection.connected.copy()
    for label in np.unique(connection.labels[connection.labels>=0]):
        mask=connection.labels==label
        mean_gap=np.trunc(observed[mask].mean())
        ref=np.trunc(average+np.copysign(.5,average))
        observed[mask]+=float(align_period(np.array([mean_gap]),ref,PERIOD)[0]-mean_gap)
    connected=np.isfinite(observed)
    fill=np.trunc(observed[connected].mean()) if connected.any() else np.trunc(average)
    observed[valid&~connected]=fill
    good_merit=np.isfinite(connection.merit)&valid
    maximum=float(connection.merit[good_merit].max()) if good_merit.any() else 0
    if maximum<=0:raise ArithmeticError('High 2G 连接质量归一化奇异。')
    weights=np.where(good_merit,1-.9*connection.merit/maximum,.1)
    y,x=np.indices(valid.shape);design=np.column_stack((np.ones(valid.sum()),x[valid],y[valid],filtered[valid]))
    w=weights[valid];target=observed[valid]
    coefficients=np.linalg.solve(design.T@(w[:,None]*design),design.T@(w*target))
    model=design@coefficients
    q=np.trunc(w*PERIOD)/PERIOD
    gap=np.trunc(model+w.mean()*q*q*(target-model))
    output=np.full(valid.shape,np.nan)
    output[valid]=align_period(fine[valid],filtered[valid]+gap,PERIOD)/scale
    final_gap=np.full(valid.shape,np.nan);final_gap[valid]=gap*2*np.pi/PERIOD
    return output,{'high2g_raw':np.where(valid,raw_gap*2*np.pi/PERIOD,np.nan),
                   'high2g_final':final_gap,'high2g_connected':observed*2*np.pi/PERIOD,
                   'merit_map':connection.merit,'confidence_map':np.where(valid,weights,0),
                   'region_labels':connection.labels,'connected_mask':connection.labels>=0,
                   'theta_map_smoothed':np.where(valid,filtered*2*np.pi/PERIOD,np.nan),
                   'final_height_phase_map':output*k,'high2g_coefficients':coefficients}
