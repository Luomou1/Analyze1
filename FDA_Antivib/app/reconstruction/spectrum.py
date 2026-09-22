"""扫描数据公共前端，按实际帧数扩展 EXE 的频域拟合结构。

EXE 0xAA4430..0xAA5060 对有效连续区间从左向右展开并用功率加权；
不调用旧项目的峰值双向解包裹、lg、DLL级次处理或GFDA估计器。
"""
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray

FloatArray=NDArray[np.float64]


def prepare_signals(block: FloatArray,bins: NDArray[np.int64]):
    """根据每像素载频估计半周期差分峰，循环移动实际N帧到记录中部。"""
    n=block.shape[1]
    initial=np.abs(np.fft.rfft(block-block.mean(1,keepdims=True),axis=1))[:,bins]
    carrier_bins=bins[np.argmax(initial,axis=1)]
    lag=np.maximum(1,np.floor(n/(2*carrier_bins)+.5).astype(int))
    peak_frames=np.empty(len(block),dtype=int)
    for delay in np.unique(lag):
        selected=lag==delay
        peak_frames[selected]=np.argmax(block[selected,delay:]-block[selected,:-delay],axis=1)+delay
    sample_indices=(np.arange(n)[None,:]+peak_frames[:,None]-n//2)%n
    return np.take_along_axis(block,sample_indices,axis=1),peak_frames


@dataclass(frozen=True,slots=True)
class SpectrumResult:
    coarse_um: FloatArray
    phase: FloatArray
    slope: FloatArray
    intercept: FloatArray
    peak_bin: FloatArray
    power: FloatArray
    valid: NDArray[np.bool_]
    bin_start: NDArray[np.int64]
    bin_end: NDArray[np.int64]
    reference_um: FloatArray
    frame_count: int
    step_um: float


def phase_fit(spectra: NDArray[np.complex128], bins: NDArray[np.int64], threshold: float, unwrap_method="exe", window_size=None):
    """主峰连续功率带；相位自左边界起单次±2π展开，保留float32拟合。"""
    if unwrap_method not in {"exe", "unwrap1"}:
        raise ValueError("未知解包裹方式")
    if window_size is not None and (not isinstance(window_size, int) or window_size < 1):
        raise ValueError("左右频点数必须为正整数")
    power=(spectra.real**2+spectra.imag**2).astype(np.float32)
    phase=np.angle(spectra*(-1.)**bins).astype(np.float32)
    good=np.isfinite(power)&(power>threshold)
    row=np.arange(len(power));peak=np.argmax(np.where(good,power,0),axis=1)
    left=peak.copy();right=peak.copy()
    peak_valid=good[row,peak].copy()
    if window_size is not None:
        # 用户窗口按频点编号选取，不再由功率阈值扩张；零功率点仍不可拟合。
        good=np.isfinite(power)&(power>0)
    for direction,edge in [(-1,left),(1,right)]:
        alive=peak_valid.copy()
        extent=len(bins)-1 if window_size is None else min(window_size,len(bins)-1)
        for d in range(1,extent+1):
            col=peak+direction*d
            alive&=(col>=0)&(col<len(bins))
            rows=row[alive]
            alive[rows[~good[rows,col[rows]]]]=False
            rows=row[alive];edge[rows]=col[rows]
    valid=peak_valid&(right-left>=2)&(peak>left)&(peak<right)
    unwrapped=np.full_like(phase,np.nan)
    if unwrap_method == "unwrap1":
        unwrapped[row[valid],peak[valid]]=phase[row[valid],peak[valid]]
        for direction in (-1,1):
            for d in range(1,len(bins)):
                col=peak+direction*d
                selected=valid&(col>=left)&(col<=right)
                rows=row[selected];cols=col[selected]
                delta=phase[rows,cols]-unwrapped[rows,cols-direction]
                turns=np.where(delta>np.pi,np.ceil((delta-np.pi)/(2*np.pi)),
                               np.where(delta < -np.pi,np.floor((delta+np.pi)/(2*np.pi)),0))
                unwrapped[rows,cols]=phase[rows,cols]-turns*(2*np.pi)
        # 先恢复频谱参考再拟合；调用高度分支时显式换回其中心化系数约定。
        unwrapped=unwrapped.astype(float)-np.pi*bins[None,:]
    else:
        for j in range(len(bins)):
            selected=(left<=j)&(right>=j)&valid
            starts=selected&(left==j)
            unwrapped[starts,j]=phase[starts,j]
            if j:
                selected&=left<j
                diff=phase[selected,j]-unwrapped[selected,j-1]
                unwrapped[selected,j]=phase[selected,j]+np.where(diff>np.pi,-2*np.pi,np.where(diff < -np.pi,2*np.pi,0))
    weights=np.where(np.isfinite(unwrapped),power,0)
    x=(bins[None,:]-bins[left,None]).astype(np.float32)
    if unwrap_method == "unwrap1":
        weights=weights.astype(float);x=x.astype(float)
    y=np.nan_to_num(unwrapped,nan=0)
    s0=weights.sum(1);s1=(weights*x).sum(1);s2=(weights*x*x).sum(1)
    t0=(weights*y).sum(1);t1=(weights*x*y).sum(1)
    determinant=s0*s2-s1*s1
    valid&=determinant>0
    slope=np.full(len(row),np.nan,dtype=np.float64 if unwrap_method=="unwrap1" else np.float32);local=np.full_like(slope,np.nan)
    np.divide(s0*t1-s1*t0,determinant,out=slope,where=valid)
    np.divide(s2*t0-s1*t1,determinant,out=local,where=valid)
    intercept=local-slope*bins[left]
    peak_fraction=np.full(len(row),np.nan)
    rows=row[valid];cols=peak[rows]
    denominator=2*(power[rows,cols-1]+power[rows,cols+1]-2*power[rows,cols])
    nonzero=denominator!=0;valid[rows[~nonzero]]=False
    rows=rows[nonzero];cols=cols[nonzero]
    peak_fraction[rows]=bins[cols]+(power[rows,cols-1]-power[rows,cols+1])/denominator[nonzero]
    intercept[~valid]=np.nan;slope[~valid]=np.nan
    return slope.astype(float),intercept.astype(float),peak_fraction,valid,bins[left],bins[right],unwrapped,power


def reconstruct_spectrum(cube: np.ndarray,step_um: float,start_um: float=0,
                         maximum: float=4095,modulation: float=7,
                         progress: Callable[[int],None]|None=None, unwrap_method="exe", window_size=3) -> SpectrumResult:
    """输入实际N帧，输出正扫描物理参考下的粗高度和逐像素相位参数。"""
    if cube.ndim!=3 or cube.shape[2]<10 or step_um<=0 or not np.isfinite(step_um):
        raise ValueError('需要至少10帧干涉图和正的扫描步长。')
    if not np.isfinite(cube).all() or maximum<=0 or not 0<=modulation<=100:
        raise ValueError('强度或调制度参数无效。')
    shape=cube.shape[:2];n=cube.shape[2];flat=cube.reshape(-1,n)
    bins=np.arange(2,n//2,dtype=np.int64)
    # 实际位深浮点输入使用明确的绝对DFT功率门限；不冒充EXE打包8bit缩放。
    threshold=16*(modulation*maximum/100)**2
    slope,intercept,peaks,powers=(np.full(len(flat),np.nan) for _ in range(4))
    references=np.zeros(len(flat))
    valid=np.zeros(len(flat),dtype=bool);left=np.zeros(len(flat),dtype=np.int64);right=left.copy()
    for start in range(0,len(flat),2048):
        end=min(len(flat),start+2048);block=flat[start:end].astype(float)
        # 保留实际N帧，用循环移位模拟原厂峰帧参考；不截64帧、不补造样本。
        shifted,peak_frames=prepare_signals(block,bins)
        spectrum=np.conj(np.fft.rfft(shifted,axis=1))[:,bins]
        b,a,p,ok,l,r,_,w=phase_fit(spectrum,bins,threshold,unwrap_method,window_size)
        if unwrap_method == "unwrap1":
            b=b+np.pi
        ok&=~np.any(block>=maximum,axis=1)
        slope[start:end]=np.where(ok,b,np.nan);intercept[start:end]=np.where(ok,a,np.nan)
        peaks[start:end]=np.where(ok,p,np.nan);powers[start:end]=np.where(ok,w.max(1),np.nan)
        valid[start:end]=ok;left[start:end]=l;right[start:end]=r
        references[start:end]=start_um+(peak_frames+n/2-n//2)*step_um
        if progress:progress(int(70*end/len(flat)))
    coarse=references+slope*n*step_um/(2*np.pi)
    phase=intercept+slope*peaks
    arrays=[a.reshape(shape) for a in [coarse,phase,slope,intercept,peaks,powers,valid,left,right]]
    return SpectrumResult(*arrays,references.reshape(shape),n,step_um)
