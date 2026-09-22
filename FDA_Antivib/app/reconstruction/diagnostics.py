"""像素诊断直接使用新重建层的同一频谱、有效带和拟合。"""
import numpy as np
from app.reconstruction.spectrum import phase_fit,prepare_signals


def build_pixel_analysis(intensity_data,x,y,step_size,start_height=0.,maximum_scan_value=4095.,**options):
    raw=np.asarray(intensity_data[y,x],dtype=float);n=len(raw);bins=np.arange(2,n//2)
    shifted,peak_frame=prepare_signals(raw[None],bins)
    spectrum=np.conj(np.fft.rfft(shifted,axis=1))
    b,a,peak,valid,_,_,phase,power=phase_fit(spectrum[:,bins],bins,16*(7*maximum_scan_value/100)**2,options.get("unwrap_method","exe"),options.get("window_size",3))
    if options.get("unwrap_method", "exe") == "unwrap1":
        # 撤销输入循环移位的线性相位，显示原采样序列参考。
        shift_slope=2*np.pi*(peak_frame[0]-n//2)/n
        phase=phase+shift_slope*bins[None,:]
        b=b+shift_slope
    good=np.isfinite(phase[0])&valid[0]
    if np.any(raw>=maximum_scan_value):good[:]=False
    k=np.arange(spectrum.shape[1])*np.pi/(n*step_size)
    amplitude=np.abs(spectrum[0])*2/n;amplitude[0]=0
    k0=float(peak[0]*np.pi/(n*step_size)) if valid[0] else float(k[2])
    # 全频段展示与有效拟合带分开；弱频点可显示，但不加入拟合。
    wrapped=np.angle(spectrum[0]*(-1.)**np.arange(len(k)))
    wrapped[np.abs(spectrum[0])==0]=np.nan
    original=np.conj(np.fft.rfft(raw-raw.mean()))
    original_phase=np.angle(original)
    original_phase[np.abs(original)==0]=np.nan
    original_phase[0]=np.nan
    unwrapped=np.full(len(k),np.nan);unwrapped[bins[good]]=phase[0,good]
    return {'x':x,'y':y,'signal_x':start_height+np.arange(n)*step_size,'signal_raw_y':raw,
            'signal_dc_y':raw-raw.mean(),'window_y':np.ones(n),'signal_windowed_y':raw,
            'k_x':k,'amplitude_y':amplitude,'k0_x':k0,'k0_y':float(np.interp(k0,k,amplitude)),
            'phase_original_y':original_phase,'fit_point_count':int(good.sum()),'phase_raw_y':wrapped,'phase_unwrapped_y':unwrapped,'fit_mask_k_x':k[bins[good]],
            'fit_mask_phase_y':phase[0,good],'fit_k_x':k[bins[good]],'fit_phase_y':a[0]+b[0]*bins[good],
            'fit_valid':bool(good.any()),'fft_length':n,'k0_index':int(round(peak[0])) if valid[0] else -1,
            'reference_origin_um':start_height+(peak_frame[0]+n/2-n//2)*step_size}
