import os
from ark.running import complete
from ark.data import load
from ark.setting import *
from ark.nn.text_process import Vocab, fusion_piny_letter
from ark.nn.module import AttentionArk
from ark.nn.valid import k_fold_valid
from ark.nn.accuracy import save_fig


@complete
def main():
    train_texts, train_labels, test_texts, test_labels = load.load_train_test_data(-1, drop_test=True)

    vocab = Vocab(VOCAB_PATH)

    text_layer = fusion_piny_letter

    train_x, valid_len = text_layer(train_texts, vocabs=vocab, steps=128, front_pad=True)

    hidden_size, num_heads, en_layer, de_layer = 64, 4, 4, 8
    ark = AttentionArk(vocab, hidden_size=hidden_size, in_channel=3, num_steps=128, num_heads=num_heads,
                       en_num_layer=en_layer, de_num_layer=de_layer,
                       dropout=0.5, num_class=2)

    optim_params = {'lr': 1e-3, 'weight_decay': 1e-3}
    k_loss_list, k_train_acc, k_valid_acc, ark = k_fold_valid(5, train_x, train_labels, valid_len, model=ark,
                                                              num_class=2, num_valid=-1, batch_size=128,
                                                              epochs=200, stop_loss_value=1, stop_min_epoch=10,
                                                              optim_params=optim_params)

    avg_acc = 0
    for i, valid_acc in enumerate(k_valid_acc):
        avg_acc += valid_acc.max().score
        valid_acc.plot('epochs', 'accuracy', [f'fold-{i}'], 'valid-k-fold-cross-valid', save=False)
    save_fig('valid.png')

    for i, train_acc in enumerate(k_train_acc):
        train_acc.plot('epochs', 'accuracy', [f'fold-{i}'], 'train-k-fold-cross-valid', save=False)
    save_fig('train.png')

    print('avg acc:', avg_acc / len(k_valid_acc))
    for sub_ark, sub_acc in zip(ark, k_valid_acc):
        path = os.path.join(MODEL_LIB,
                            f'ark-{sub_acc.max().score: .2f}-{hidden_size}-{num_heads}-{en_layer}-{de_layer}.net')
        sub_ark.save_state_dict(path)


main()
