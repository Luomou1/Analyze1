"""固定工件必须与来源元数据匹配；运行测试不加载原厂 DLL。"""
import hashlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('name', ['poly4', 'fft', 'spline', 'robust_spline', 'regions', 'sinusoid','sphere','cone','filter_control', 'fill_boundaries', 'chain', 'zernike', 'differential', 'exe_e2e'])
def test_surface_fixture_matches_recorded_hash(name: str) -> None:
    path = Path(__file__).parent / 'data' / f'surface_{name}_native.npz'
    metadata = json.loads(path.with_suffix('.npz.json').read_text(encoding='utf-8'))
    assert metadata['fixture_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    expected = ('ab2b7f1063b5485e7a02c9ff131edfc4459b4023926d094d8d9fa7a8bfcfaefd' if name in ('filter_control', 'exe_e2e') else
                '440b0d068ff74489373d4744ddeba71604092807c8500e18d0e864e3b1d55c0d')
    assert metadata['source_sha256'] == expected
