import collections
import datetime
import enum
import os
import pathlib
import pickle
import typing as tp

import numpy as np
import pandas as pd
import torch
import tqdm.autonotebook
import transformers
import trl
import wandb

import metrics
import models
import wrappers


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}')

if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    import kaggle_secrets
    CHECKPOINT_PATH = pathlib.Path('../input/nl2ml-generator/checkpoints/AttentiveModel-2023-04-28.pt')
    CODE4ML_DIR = pathlib.Path('../input/code4ml')
    TRAIN_DATA_DIR = pathlib.Path('../input/nl2ml-generator/data')
    WANDB_KEY = kaggle_secrets.UserSecretsClient().get_secret('wandb_key')
else:
    CHECKPOINT_PATH = pathlib.Path('generator/checkpoints/AttentiveModel-2023-04-28.pt')
    CODE4ML_DIR = pathlib.Path('code4ml')
    TRAIN_DATA_DIR = pathlib.Path('generator/data')
    WANDB_KEY = os.environ.get('WANDB_KEY')

TIMEZONE = datetime.timezone(datetime.timedelta(hours=3))
WANDB_PROJECT_NAME = 'rl4nl2ml'


class ResponsePadder(transformers.PreTrainedTokenizer):
    """
    Only used to pad the response tensors (we pad the query tensors manually in ``wrappers.flatten_queries``).
    Passed to ``trl.PPOTrainer`` as a "tokenizer" (because it requires one) but doesn't actually tokenize anything.
    """
    pad_token = 'pad'    # unused but needs to be set to something otherwise it throws an error
    pad_token_id = models.PAD_ID


def fix_ppo_model_state_dict(state_dict: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """
    Fixes a state dict obtained with ``trl.AutoModelForSeq2SeqLMWithValueHead.state_dict()``
    so that it can then be passed to ``trl.AutoModelForSeq2SeqLMWithValueHead.load_state_dict``.
    (If you just pass it directly it throws an error because some of the keys are wrong.)

    :param state_dict: a state dict for a ``trl.AutoModelForSeq2SeqLMWithValueHead``
    :return: a fixed state dict suitable for loading
    """
    return {f'pretrained_model.{key}' if key.startswith('model.') else key: value for key, value in state_dict.items()}


class Reward(str, enum.Enum):
    """An enum of the currently implemented reward types. They have string representations for easier wandb logging."""
    BLEU_MINWER = 'bleu - minwer'
    BLEU_MINWER_SQ_LEN_DIFF = 'bleu - min(wer + len_diff ** 2)'

    def __str__(self):
        return self.value


cfg = {
    # Model parameters
    'model': 'AttentiveModel',
    'checkpoint': '2023-04-28',
    'n_tokens': 78,
    'enc_size': 768,
    'dec_size': 788,
    'emb_size': 128,
    'attn_size': 128,

    # Reward parameters
    'reward': Reward.BLEU_MINWER,
    'bleu_coef': 1,
    'minwer_coef': 0,

    # PPO parameters
    'batch_size': 256,
    'mini_batch_size': 128,
    'adap_kl_ctrl': True,
    'init_kl_coef': 0.2,
    'horizon': 10000,

    # Optimizer parameters
    'optimizer': 'adam',
    'init_lr': 1e-4,

    # Scheduler parameters
    'scheduler': 'multistep',
    'multistep_epochs': [2],
    'multistep_gamma': 0.2,

    # Inference parameters
    'train_temperature': 1,
    'test_temperature': 0,

    # Metric parameters
    'bleu_weights': [0.5, 0.5],

    # Stopping parameters
    'n_epochs': 10,
    'early_stopping_kl_threshold': -0.5,
    'early_stopping_patience': 3,

    # Reproducibility
    'random_seed': 27,
}

log_wandb: bool = False
resume_wandb_run_id: str | None = None
resume_checkpoint_path: str | None = None

trl.set_seed(cfg['random_seed'])    # seeds builtin, numpy and torch random

graph = pd.read_csv(CODE4ML_DIR / 'vertices.csv')
with open(TRAIN_DATA_DIR / 'train_untrained_bert_tabular.p', 'rb') as fp:
    trainval_data = pickle.load(fp)
with open(TRAIN_DATA_DIR / 'test_untrained_bert_tabular.p', 'rb') as fp:
    test_data = pickle.load(fp)
with open(TRAIN_DATA_DIR / 'test_comps_tabular.p', 'rb') as fp:
    test_comps = pickle.load(fp)
with open(TRAIN_DATA_DIR / 'val_kernel_ids_tabular.p', 'rb') as fp:
    val_kernel_ids = pickle.load(fp)

for sample in trainval_data + test_data:
    sample['h_0'] = [sample['encoded_sequence'], sample['encoded_tokens']]
    del sample['encoded_sequence']
    del sample['encoded_tokens']

train_data = []
val_data = []
val_comps = set()
for sample in trainval_data:
    if sample['kernel_id'] in val_kernel_ids:
        val_data.append(sample)
        val_comps.add(sample['comp_name'])
    else:
        train_data.append(sample)

print(f'validation samples: {len(val_data)}')
print(f'training samples: {len(train_data)}')
n_batches_per_epoch = (len(train_data) - 1) // cfg['batch_size'] + 1

train_references = collections.defaultdict(list)
for sample in train_data:
    train_references[sample['comp_name']].append([str(token) for token in sample['target'].tolist()])
train_mean_target_length = {comp: sum(len(ref) for ref in refs) / len(refs) for comp, refs in train_references.items()}

model_cfg = wrappers.AttentiveModelConfig(
    n_tokens=cfg['n_tokens'],
    enc_size=cfg['enc_size'],
    dec_size=cfg['dec_size'],
    emb_size=cfg['emb_size'],
    attn_size=cfg['attn_size'],
    device=DEVICE,
    hidden_size=cfg['enc_size']    # passed through to the trl ValueHead constructor
)
pretrained_state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
wrapper = wrappers.AttentiveModelWrapper(model_cfg)
wrapper.model.load_state_dict(pretrained_state)
wrapper.to(DEVICE)

trl.AutoModelForSeq2SeqLMWithValueHead.lm_head_namings.append('logits')
ppo_model = trl.AutoModelForSeq2SeqLMWithValueHead.from_pretrained(wrapper)

ppo_cfg = trl.PPOConfig(
    adap_kl_ctrl=cfg['adap_kl_ctrl'],
    init_kl_coef=cfg['init_kl_coef'],
    horizon=cfg['horizon'],
    seed=cfg['random_seed'] + 1,    # PPOTrainer always calls trl.set_seed on creation
    # batch_size and mini_batch_size are set manually before every ppo_trainer.step(...) call,
    # because batches can vary in size
)
optimizer = torch.optim.Adam(ppo_model.parameters(), lr=cfg['init_lr'])
milestones = [epoch * n_batches_per_epoch for epoch in cfg['multistep_epochs']]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=cfg['multistep_gamma'])
ppo_trainer = trl.PPOTrainer(ppo_cfg, ppo_model, optimizer=optimizer, lr_scheduler=scheduler, tokenizer=ResponsePadder())

if resume_checkpoint_path is not None:
    checkpoint = torch.load(resume_checkpoint_path)
    ppo_model.load_state_dict(fix_ppo_model_state_dict(checkpoint['model']))
    ppo_trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    if checkpoint.get('scheduler') is not None:
        ppo_trainer.lr_scheduler.load_state_dict(checkpoint['scheduler'])
    ppo_trainer.kl_ctl = pickle.loads(checkpoint['kl_ctl'])

    first_epoch = checkpoint['epoch']
    first_batch = checkpoint['batch'] + 1
    max_val_bleu = checkpoint['max_val_bleu']
    min_val_minwer = checkpoint['min_val_minwer']
    log = checkpoint['log']
else:
    first_epoch = 1
    first_batch = 1
    max_val_bleu = -1
    min_val_minwer = 1e9
    log = []

val_mtx = metrics.compute_metrics(
    wrapper.model,
    val_comps,
    val_data,
    cfg['test_temperature'],
    cfg['bleu_weights']
)

test_mtx = metrics.compute_metrics(
    wrapper.model,
    test_comps,
    test_data,
    cfg['test_temperature'],
    cfg['bleu_weights']
)

print('Initial model stats')
for metric, score in val_mtx.items():
    print(f'val {metric}: {score:.3f}')
for metric, score in test_mtx.items():
    print(f'test {metric}: {score:.3f}')
print()

if log_wandb:
    wandb.login(anonymous='never', key=WANDB_KEY)
    if resume_wandb_run_id is None:
        wandb.init(project=WANDB_PROJECT_NAME, config=cfg, save_code=True)
    else:
        wandb.init(project=WANDB_PROJECT_NAME, config=cfg, save_code=True, id=resume_wandb_run_id, resume='must')
    run_string, run_number = wandb.run.name.rsplit('-', maxsplit=1)
    run_name = f'{run_number.zfill(2)}-{run_string}'
else:
    run_name = datetime.datetime.now(TIMEZONE).strftime('local-%m-%d-%H%M')

early_stopping_patience = cfg['early_stopping_patience']
trl.set_seed(cfg['random_seed'] + 2)    # seeds builtin, numpy and torch random
for epoch in range(first_epoch, cfg['n_epochs'] + 1):
    for batch in range(first_batch if epoch == first_epoch else 1, n_batches_per_epoch + 1):
        print(f'Epoch {epoch}/{cfg["n_epochs"]}, batch {batch}/{n_batches_per_epoch}')

        queries = []
        responses = []
        rewards = []
        samplewise_stats = collections.defaultdict(list)
        samples = train_data[(batch - 1) * cfg['batch_size'] : batch * cfg['batch_size']]

        for sample in tqdm.autonotebook.tqdm(samples, desc='generating'):
            query = sample['h_0']
            response = wrapper.model.generate(query, temperature=cfg['train_temperature'], return_tokens=True)
            if len(response) < 4:
                response.extend([models.PAD_ID] * (4 - len(response)))    # trl requirement
            str_response = [str(token) for token in response]

            len_diff = len(response) - train_mean_target_length[sample['comp_name']]
            repeats_portion = 1 - len(set(response)) / len(response)

            samplewise_stats['env/len_diff'].append(len_diff)
            samplewise_stats['env/abs_len_diff'].append(abs(len_diff))
            samplewise_stats['env/repeats_portion'].append(repeats_portion)

            if cfg['reward'] == Reward.BLEU_MINWER:
                reward = 0
                if cfg['bleu_coef'] != 0:
                    response_bleu = metrics.bleu([str_response], [train_references[sample['comp_name']]], cfg['bleu_weights'])
                    samplewise_stats['env/bleu'].append(response_bleu)
                    reward += cfg['bleu_coef'] * response_bleu
                if cfg['minwer_coef'] != 0:
                    response_minwer = min(metrics.wer([str_response], [ref]) for ref in train_references[sample['comp_name']])
                    samplewise_stats['env/minwer'].append(response_minwer)
                    reward -= cfg['minwer_coef'] * response_minwer
            elif cfg['reward'] == Reward.BLEU_MINWER_SQ_LEN_DIFF:
                reward = 0
                if cfg['bleu_coef'] != 0:
                    response_bleu = metrics.bleu([str_response], [train_references[sample['comp_name']]], cfg['bleu_weights'])
                    samplewise_stats['env/bleu'].append(response_bleu)
                    reward += cfg['bleu_coef'] * response_bleu
                if cfg['minwer_coef'] != 0:
                    response_minwer_sqld = min(
                        metrics.wer([str_response], [ref]) + cfg['len_diff_coef'] * ((len(response) - len(ref)) ** 2)
                        for ref in train_references[sample['comp_name']]
                    )
                    samplewise_stats['env/minwer_sqld'].append(response_minwer_sqld)
                    reward -= cfg['minwer_coef'] * response_minwer_sqld
            else:
                raise NotImplementedError

            queries.append(query)
            responses.append(torch.as_tensor(response, dtype=torch.int64))
            rewards.append(torch.as_tensor(reward, dtype=torch.float))

        samplewise_stats['env/reward'] = rewards

        print('PPO step... ', end='')
        ppo_cfg.batch_size = len(queries)
        if len(queries) % cfg['mini_batch_size'] == 0:
            ppo_cfg.mini_batch_size = cfg['mini_batch_size']
        else:
            ppo_cfg.mini_batch_size = len(queries)
        # noinspection PyTypeChecker
        stats = ppo_trainer.step(wrappers.flatten_queries(queries), responses, rewards)
        print('done')

        val_mtx = metrics.compute_metrics(
            wrapper.model,
            val_comps,
            val_data,
            cfg['test_temperature'],
            cfg['bleu_weights']
        )
        for metric, score in val_mtx.items():
            print(f'val {metric}: {score:.3f}')
            stats[f'val/{metric}'] = score
        print()

        for stat, values in samplewise_stats.items():
            stats.update({
                f'{stat}_mean': np.mean(values),
                f'{stat}_std': np.std(values),
                f'{stat}_dist': wandb.Histogram(values)
            })

        stats['epoch'] = epoch
        log.append(stats)
        if log_wandb:
            wandb.log(stats)

        checkpoint = {
            'model': ppo_model.state_dict(),
            'optimizer': ppo_trainer.optimizer.state_dict(),
            'scheduler': ppo_trainer.lr_scheduler.state_dict() if ppo_trainer.lr_scheduler is not None else None,
            'kl_ctl': pickle.dumps(ppo_trainer.kl_ctl),
            'epoch': epoch,
            'batch': batch,
            'max_val_bleu': max_val_bleu,
            'min_val_minwer': min_val_minwer,
            'config': cfg,
            'log': log
        }

        suffixes = ['last']
        if val_mtx['bleu'] > max_val_bleu:
            max_val_bleu = val_mtx['bleu']
            suffixes.append('maxbleu')
        if val_mtx['minwer'] < min_val_minwer:
            min_val_minwer = val_mtx['minwer']
            suffixes.append('minminwer')

        for suffix in suffixes:
            path = f'ckpt-{run_name}-{suffix}.pt'
            torch.save(checkpoint, path)
            print(f'Checkpoint saved to {path}')
        print()

        if stats['objective/kl'] < cfg['early_stopping_kl_threshold']:
            # Sometimes the KL divergence computed by trl can go wildly into the negatives.
            # It's unclear exactly why this happens, but when it does it seems to make the model worse,
            # so we stop early if the KL divergence is too negative for too many steps.
            # See also https://github.com/lvwerra/trl/issues/235
            early_stopping_patience -= 1
            print(f'KL < {cfg["early_stopping_kl_threshold"]}! Patience = {early_stopping_patience}\n')
            if early_stopping_patience == 0:
                break
    if early_stopping_patience == 0:
        break

for suffix in ['last', 'maxbleu', 'minminwer']:
    checkpoint = torch.load(f'ckpt-{run_name}-{suffix}.pt')
    ppo_model.load_state_dict(fix_ppo_model_state_dict(checkpoint['model']))

    print(f'{run_name}-{suffix} (epoch {checkpoint["epoch"]}, batch {checkpoint["batch"]})')
    if log_wandb:
        wandb.run.summary[f'summary/{suffix}/epoch'] = checkpoint['epoch']
        wandb.run.summary[f'summary/{suffix}/batch'] = checkpoint['batch']

    val_mtx = metrics.compute_metrics(
        wrapper.model,
        val_comps,
        val_data,
        cfg['test_temperature'],
        cfg['bleu_weights']
    )

    test_mtx = metrics.compute_metrics(
        wrapper.model,
        test_comps,
        test_data,
        cfg['test_temperature'],
        cfg['bleu_weights']
    )

    for metric, score in val_mtx.items():
        print(f' val {metric}: {score:.3f}')
        if log_wandb:
            wandb.run.summary[f'summary/{suffix}/val/{metric}'] = score
    for metric, score in test_mtx.items():
        print(f' test {metric}: {score:.3f}')
        if log_wandb:
            wandb.run.summary[f'summary/{suffix}/test/{metric}'] = score

    print(' generation examples:')
    for i, sample in enumerate(test_data[:10]):
        print(f'  {i + 1}. {wrapper.model.generate(sample["h_0"], graph=graph, temperature=cfg["test_temperature"])}')
    print()

if log_wandb:
    wandb.finish()
