import torch
import numpy as np
from igcrn.model import IGCRN, spectral_loss
from scripts.simulate import rir
from scripts.evaluate import sisdr

torch.set_num_threads(2)
def test_shape_gradients():
    torch.manual_seed(2026); m=IGCRN(width=8,depth=2); x=torch.randn(2,2,1024)*.1
    z=m.spectrum(x); assert z.shape==(2,257,5)
    loss=spectral_loss(z,m.stft(x[:,0])); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m(x).shape==(2,1024)

def test_reconstruction():
    m=IGCRN(); x=torch.randn(2,3587)
    assert torch.allclose(m.istft(m.stft(x),x.shape[-1]),x,atol=1e-6)

def test_frequency_preserved_and_reused():
    m=IGCRN(width=8,depth=2); shapes=[]
    h=m.rnn.register_forward_pre_hook(lambda module,args:shapes.append(args[0].shape))
    m(torch.randn(2,2,1024)); h.remove()
    assert shapes==[torch.Size([2*257,5,8])]

def test_metrics_and_rir():
    rng=np.random.default_rng(3); x=rng.normal(size=2000)
    assert abs(sisdr(2*x,x)-sisdr(x,x))<7 # epsilon affects perfect reconstruction
    assert sisdr(x+.1*rng.normal(size=2000),x)>15
    a=rir([2,4,1.5],np.array([2.49,2.5,1.5])); b=rir([2,4,1.5],np.array([2.51,2.5,1.5]))
    assert np.array_equal(a,rir([2,4,1.5],np.array([2.49,2.5,1.5]))) and not np.array_equal(a,b)

def test_checkpoint_roundtrip(tmp_path):
    m=IGCRN(width=8,depth=2).eval(); x=torch.randn(1,2,1024)
    torch.save(m.state_dict(),tmp_path/'m.pt'); n=IGCRN(width=8,depth=2).eval()
    n.load_state_dict(torch.load(tmp_path/'m.pt',weights_only=True))
    assert torch.equal(m(x),n(x))
