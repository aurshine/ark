from typing import List

from . import train_module
from .data.dataloader import get_ark_loader
from .data.load import load
from .setting import *
from .train_module import train, pre_train
from .utils import *


# Ark模型
__ark__ = None
# Ark保存模型路径
__ark_pth__ = os.path.join(os.path.dirname(__file__), 'ark5.pth')
# Ark运行设备
__ark_device__ = use_device()

def default_ark():
    """
    构造默认的Ark模型
    """
    global __ark_pth__, __ark_device__

    ark_classifier = train_module.ArkClassifier(hidden_size=train_module.HIDDEN_SIZE,
                                                num_classes=train_module.NUM_CLASS,
                                                num_heads=train_module.NUM_HEADS,
                                                dropout=train_module.DROPOUT,
                                                device=use_device(__ark_device__)
                                                )
    # 构造Ark模型
    dft_ark = train_module.Ark(tokenizer=train_module.TOKENIZER,
                               output_layer=ark_classifier,
                               steps=train_module.STEPS,
                               hidden_size=train_module.HIDDEN_SIZE,
                               num_heads=train_module.NUM_HEADS,
                               num_layer=train_module.NUM_LAYER,
                               num_channels=3,
                               dropout=train_module.DROPOUT,
                               num_class=train_module.NUM_CLASS,
                               device=use_device(__ark_device__),
                               prefix_name='user_ark'
                            )
    if __ark_pth__ is not None:
        dft_ark.load(__ark_pth__)
        
    return dft_ark

def _init_ark(ark_pth: str = None, device=None):
    global __ark__, __ark_pth__, __ark_device__

    if __ark__ is None:
        __ark__ = default_ark()
    
    if ark_pth != __ark_pth__:
        __ark__.load(ark_pth)
        __ark_pth__ = ark_pth
    
    if device != __ark_device__:
        __ark__.to(device)
        __ark_device__ = device
    
def ark(texts: Union[str, List[str]],
        batch_size: int = train_module.BATCH_SIZE,
        ark_pth: str = None,
        device=None
        ):
    """
    预测文本的情感倾向, 0表示非恶意, 1表示恶意

    :param texts: 输入文本列表

    :param batch_size: 批处理大小
    
    :param ark_pth: Ark模型的路径,如果为None则使用默认模型

    :param device: 运行设备,默认为None,即自动选择运行设备

    :return: 输入文本的情感倾向
    """
    global __ark__

    device = use_device(device)
    _init_ark(ark_pth or __ark_pth__, device or __ark_device__)

    if isinstance(texts, str):
        texts = [texts]
    
    ark_loader = get_ark_loader(texts=texts, 
                                labels=None, 
                                tokenizer=__ark__.tokenizer,
                                shuffle=False,
                                max_length=train_module.STEPS,
                                batch_size=batch_size, 
                                device=device
                            )
    
    results = []
    for data in ark_loader:
        result = __ark__.predict([data['source_tokens']['input_ids'], 
                                  data['initial_tokens']['input_ids'], 
                                  data['final_tokens']['input_ids']],
                                data['source_tokens']['attention_mask']
                                )
        results.append(result)
    
    return torch.cat(results).detach().cpu()
                         