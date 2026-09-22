"""EXE High 的梯度变化剔除与逐区域周期展开。"""
import numpy as np
from numpy.typing import NDArray


def reject_discontinuities(values: NDArray[np.float64],period: int=4096,percent: float=60) -> NDArray[np.float64]:
    """0xB17110：两侧一阶差折回半周期，二阶差超过门限时剔除三个点。"""
    result=values.copy()
    limit=int(period/2*(1-percent/100))
    for axis in (0,1):
        data=np.moveaxis(values,axis,0);output=np.moveaxis(result,axis,0)
        if len(data)<3:continue
        # 边界专用循环未先检查中心NoData，而是保留0x7FFFFFF8参与int32减法。
        raw=np.where(np.isfinite(data),data,0x7ffffff8).astype(np.int32)
        a=(raw[1:-1]-raw[:-2]).astype(np.int64);b=(raw[2:]-raw[1:-1]).astype(np.int64)
        a=np.where(a>period/2,a-period,np.where(a < -period/2,a+period,a))
        b=np.where(b>period/2,b-period,np.where(b < -period/2,b+period,b))
        center_valid=np.isfinite(data[1:-1])
        center_valid[:,0]=True;center_valid[:,-1]=True
        bad=np.isfinite(data[:-2])&center_valid&np.isfinite(data[2:])&(np.abs(b-a)>limit)
        for offset in range(3):
            output[offset:offset+len(bad)][bad]=np.nan
    return result


def unwrap_regions(values: NDArray[np.float64],period: int=4096):
    """EXE 0xA240C0 的水平段展开；相邻差以整数周期折回，区域由4邻接构成。"""
    h,w=values.shape;valid=np.isfinite(values)
    output=np.full(values.shape,np.nan);labels=np.full(values.shape,-1,dtype=np.int64)
    region=0;counts=[]
    for sy,sx in zip(*np.nonzero(valid)):
        if labels[sy,sx]>=0:continue
        output[sy,sx]=values[sy,sx];queue=[(int(sy),int(sx))];count=0
        while queue:
            y,x=queue.pop()
            if labels[y,x]>=0:continue
            labels[y,x]=region;count+=1
            left=x
            while left>0 and valid[y,left-1] and labels[y,left-1]<0:
                left-=1
                delta=values[y,left]-output[y,left+1]
                output[y,left]=values[y,left]-np.floor((delta+period/2)/period)*period
                labels[y,left]=region;count+=1
            right=x
            while right+1<w and valid[y,right+1] and labels[y,right+1]<0:
                right+=1
                delta=values[y,right]-output[y,right-1]
                output[y,right]=values[y,right]-np.floor((delta+period/2)/period)*period
                labels[y,right]=region;count+=1
            for yy in (y-1,y+1):
                if not 0<=yy<h:continue
                xx=left
                while xx<=right:
                    if valid[yy,xx] and labels[yy,xx]<0:
                        delta=values[yy,xx]-output[y,xx]
                        output[yy,xx]=values[yy,xx]-np.floor((delta+period/2)/period)*period
                        queue.append((yy,xx))
                        while xx<=right and valid[yy,xx] and labels[yy,xx]<0:xx+=1
                    else:xx+=1
        if count<2:
            output[labels==region]=np.nan;labels[labels==region]=-1
        else:
            counts.append(count);region+=1
    return output,labels,np.asarray(counts)
