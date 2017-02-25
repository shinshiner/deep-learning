require 'optim'
require 'xlua'

function applyFn(fn, t, t2)
    local t_ = {}
    if type(t) == "table" then
        if t2 then
            for i = 1,#t do t_[i] = applyFn(fn, t[i], t2[i]) end
        else
            for i = 1,#t do t_[i] = applyFn(fn, t[i]) end
        end
    else t_ = fn(t, t2) end
    return t_
end

function accuracy(output,label)
    if type(output) == 'table' then
        return heatmapAccuracy(output[#output],label[#output])
    else
        return heatmapAccuracy(output,label)
    end
end

paths.dofile('hourGlass.lua')
paths.dofile('evaluation.lua')

model = createModel() --test
--model = torch.load('finalModel.t7')

datas = torch.load('preprocessed_dataset/val/data.t7')
datas = datas:float()
print('datas have been prepared')

labels = torch.load('preprocessed_dataset/val/label.t7')
labels = labels:float()
print('labels have been prepared')

optfn = optim['rmsprop']
batch_size = 4
nIters = 10
nEpochs = 200
optimState = {
    learningRate = 2.5e-4,
    learningRateDecay = 0.0,
    momentum = 0.0,
    weightDecay = 0.0,
    alpha = 0.99,
    epsilon = 1e-8
}

criterion = nn.MSECriterion()
local function evalFn(x) return criterion.output, gradparam end
cudnn.fastest = true

model:training()

model = model:cuda()
criterion = criterion:cuda()

for i=1,nIters do

	print("Now Epoching:" .. i .. "    All:" .. nIters)
	

	local loss, accu = 0.0, 0.0

	for epoch=1,nEpochs do

		xlua.progress(epoch,nEpochs)
		local data = datas[{{(epoch-1)*batch_size+1,epoch*batch_size}}]
		local label = labels[{{(epoch-1)*batch_size+1,epoch*batch_size}}]

		data = applyFn(function (x) return x:cuda() end, data)
        label = applyFn(function (x) return x:cuda() end, label)

		local output = model:forward(data)
		local err = criterion:forward(output, label)

		loss = loss + err / nEpochs

		model:zeroGradParameters()
		model:backward(data, criterion:backward(output, label)) --test
		param, gradparam = model:getParameters()
		optfn(evalFn, param, optimState)

		accu = accu + accuracy(output, label) / nEpochs
	end
	print('Loss:'..loss)
	print('Accu:'..accu)

end

torch.save('finalModel.t7',model) --test
