bhfunction result = detect_active_scanning_range_sobel(inputData, varargin)
%DETECT_ACTIVE_SCANNING_RANGE_SOBEL 确定白光干涉有效扫描范围。
%   支持 png/tif/tiff/jpg/jpeg/bmp；默认自适应确定 active scanning range。

if nargin == 0
    opts = sar_parse_options();
    inputData = sar_choose_input_folder();
elseif sar_is_option_name(inputData)
    opts = sar_parse_options(inputData, varargin{:});
    inputData = sar_choose_input_folder();
else
    opts = sar_parse_options(varargin{:});
end

if isempty(inputData)
    result = sar_build_empty_result(opts, strings(1, 0), '用户取消选择文件夹');
    warning('detectActiveRange:Canceled', '%s', result.reason);
    return
end
[frames, frameNames] = sar_load_frames(inputData, opts);
if size(frames, 3) == 0
    result = sar_build_empty_result(opts, frameNames, '输入序列为空');
    return
end

sobelScores = sar_compute_scores(frames, opts);
scores = sar_compute_temporal_scores(frames, opts);
smoothScores = sar_smooth_scores(scores, opts.SmoothWindow);
[activeMask, activeRange, thresholdValue, reason, activeRanges] = sar_select_range(smoothScores, opts);

result = struct();
result.range = activeRange;
result.ranges = activeRanges;
result.startIndex = activeRange(1);
result.endIndex = activeRange(2);
result.activeMask = activeMask;
result.temporalScores = scores;
result.sobelScores = sobelScores;
result.smoothScores = smoothScores;
result.threshold = thresholdValue;
result.isValid = all(isfinite(activeRange));
result.frameCount = size(frames, 3);
result.frameNames = frameNames;
result.options = opts;
result.reason = reason;

if opts.MakePlot
    sar_plot_diagnostics(result);
end
end
