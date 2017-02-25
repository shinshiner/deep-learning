require 'torch'
require 'xlua'
require 'nn'
require 'nnx'
require 'nngraph'
require 'image'
require 'hdf5'
require 'sys'

require 'cunn'
require 'cutorch'
require 'cudnn'

nnlib = cudnn
paths.dofile('layers/Residual.lua')

nFeats = 256
nStack = 1
nModules = 1
nOutChannels = 24

local function hourglass(n, f, inp)
    local up1 = inp
    for i = 1,nModules do up1 = Residual(f,f)(up1) end

    local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)

    for i = 1,nModules do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglass(n-1,f,low1)
    else
        low2 = low1
        for i = 1,nModules do low2 = Residual(f,f)(low2) end
    end

    local low3 = low2
    for i = 1,nModules do low3 = Residual(f,f)(low3) end
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
    local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

function createModel()

    local inp = nn.Identity()()

    local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)
    local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,nFeats)(r4)

    local out = {}
    local inter = r5

    for i = 1,nStack do
        local hg = hourglass(4,256,inter)

        local ll = hg
        for j = 1,nModules do ll = Residual(nFeats,nFeats)(ll) end
        ll = lin(nFeats,nFeats,ll)

        local tmpOut = nnlib.SpatialConvolution(nFeats,nOutChannels,1,1,1,1,0,0)(ll)
        table.insert(out,tmpOut)

        if i < nStack then
            local ll_ = nnlib.SpatialConvolution(nFeats,nFeats,1,1,1,1,0,0)(ll)
            local tmpOut_ = nnlib.SpatialConvolution(nOutChannels,nFeats,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    local model = nn.gModule({inp}, out)

    return model

end
