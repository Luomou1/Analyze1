"""原厂 Sinusoid 截止传递函数：频格面积与边界弦插值。"""
import numpy as np


def sinusoid_transfer(n: int, frequency: float) -> np.ndarray:
    """构建完整频谱传递矩阵，保留原厂轴线/内部格点的不同规则。"""
    if frequency >= np.sqrt(2)*(.5+.5/n):
        return np.ones((n,n))
    r = n*frequency
    grid = np.abs(np.fft.fftfreq(n)*n)
    y, x = np.meshgrid(grid, grid, indexing='ij')
    a, b, c, d = np.maximum(x-.5, 0), x+.5, np.maximum(y-.5, 0), y+.5
    full = np.clip(np.sqrt(np.maximum(r*r-d*d, 0)), a, b)
    stop = np.clip(np.sqrt(np.maximum(r*r-c*c, 0)), a, b)
    def integral(t: np.ndarray) -> np.ndarray:
        t = np.minimum(t, r)
        return .5*(t*np.sqrt(np.maximum(r*r-t*t, 0))+r*r*np.arcsin(t/r))
    area = (full-a)*(d-c)+integral(stop)-integral(full)-c*(stop-full)
    area *= np.where(x == 0, 2, 1)*np.where(y == 0, 2, 1)
    for row, col in np.argwhere((a*a+c*c < r*r) & (b*b+d*d > r*r) & (x > 0) & (y > 0)):
        corners = [(a[row,col],c[row,col]), (b[row,col],c[row,col]),
                   (b[row,col],d[row,col]), (a[row,col],d[row,col])]
        polygon = []
        for index, start in enumerate(corners):
            end = corners[(index+1)%4]
            inside = start[0]**2+start[1]**2 <= r*r
            other_inside = end[0]**2+end[1]**2 <= r*r
            if inside:
                polygon.append(start)
            if inside != other_inside:
                if start[0] == end[0]:
                    polygon.append((start[0], np.sqrt(max(0,r*r-start[0]**2))))
                else:
                    polygon.append((np.sqrt(max(0,r*r-start[1]**2)),start[1]))
        vertices = np.asarray(polygon)
        area[row,col] = .5*abs(np.dot(vertices[:,0],np.roll(vertices[:,1],-1))-np.dot(vertices[:,1],np.roll(vertices[:,0],-1)))
    return np.clip(area, 0, 1)
