"""Normal、High、High 2G 显式调度；不保留旧FDA/GFDA/lg模式别名。"""
import numpy as np
from app.reconstruction.spectrum import reconstruct_spectrum
from app.reconstruction.high1g import reconstruct_high1g
from app.reconstruction.high import reconstruct_high
from app.reconstruction.high2g import reconstruct_high2g


def analyze(cube,step_um,mode='normal',start_um=0,maximum=4095,progress=None,unwrap_method="exe",window_size=3,common_k0=None):
    if mode not in {'normal','high','high1g','high2g'}:raise ValueError(f'未知重建模式：{mode}')
    if common_k0 is not None and (not np.isfinite(common_k0) or common_k0<=0):
        raise ValueError("手动公共K0必须为有限正数。")
    front=reconstruct_spectrum(cube,step_um,start_um,maximum,progress=progress,unwrap_method=unwrap_method,window_size=window_size)
    if not front.valid.any():raise ValueError('没有通过功率和饱和检查的有效像素。')
    match mode:
        case 'normal':height=front.coarse_um.copy();diagnostics={}
        case 'high':height,diagnostics=reconstruct_high(front)
        case 'high1g':height,diagnostics=reconstruct_high1g(front,common_k0=common_k0)
        case 'high2g':height,diagnostics=reconstruct_high2g(front,common_k0=common_k0)
    k=float(front.peak_bin[front.valid].mean())*np.pi/(front.frame_count*step_um)
    automatic_k=k
    if mode in {'high1g','high2g'} and common_k0 is not None:k=float(common_k0)
    result={'automatic_k0_value':automatic_k,'k0_source':'manual' if mode in {'high1g','high2g'} and common_k0 is not None else 'automatic','h':front.coarse_um*1000,'h_prime':height*1000,
            'phi0':front.phase,'phi0_map':front.phase,
            'heightMap':front.coarse_um*1000,'heightMap_prime':height*1000,
            'k0_value':k,'k0_index':int(round(k*front.frame_count*step_um/np.pi)),
            'fft_length':front.frame_count,'analysis_method':mode,'analysis_mode':mode,
            'theta_map':front.coarse_um*2*k,'fit_slope_map':front.slope,
            'fit_intercept_map':front.intercept,'peak_bin_map':front.peak_bin,
            'valid_band_start':front.bin_start,'valid_band_end':front.bin_end,
            'valid_mask':front.valid,'reference_origin_um':front.reference_um,
            'fringe_order_map':np.rint((height-front.coarse_um)*k/np.pi),
            'implementation_scope':'EXE分支结构，实际N帧浮点强度适配；单材料无色散标定'}
    result.update(diagnostics)
    if progress:progress(100)
    return result


def preview_spectrum(intensity_data,step_size,candidate_ratio=.1,**options):
    """公共K0使用全部有效像素；抽样频谱仅为背景显示，不决定K0。"""
    if options.get("sample_positions_um") is not None:
        raise ValueError("公共K0当前要求均匀扫描步长。")
    data=np.asarray(intensity_data);n=data.shape[-1]
    front=reconstruct_spectrum(data,step_size,
        maximum=options.get("maximum_scan_value",4095),
        unwrap_method=options.get("unwrap_method","exe"),
        window_size=options.get("window_size",3),progress=options.get("progress"))
    if not front.valid.any():
        raise ValueError("没有有效像素，无法计算公共K0。")
    common=float(front.peak_bin[front.valid].mean())*np.pi/(n*step_size)
    indices=np.flatnonzero(front.valid.ravel())
    stride=max(1,len(indices)//2048)
    sampled=data.reshape(-1,n)[indices[::stride]].astype(float)
    spectrum=np.median(np.abs(np.fft.rfft(sampled-sampled.mean(1,keepdims=True),axis=1))*2/n,axis=0)
    k=np.arange(len(spectrum))*np.pi/(n*step_size)
    value=float(np.interp(common,k,spectrum))
    return {'k_axis':k,'spectrum':spectrum,'k0_value':common,
            'peak_index':int(round(common*n*step_size/np.pi)),
            'peak_value':value,'peak_prominence':value,
            'candidate_count':int(front.valid.sum()),'fft_length':n,
            'window_name':'none','zero_padding_mode':'none'}
