"""Resume a deterministic global sample stream before distributed sharding."""

from itertools import islice

from torch.utils.data import DataLoader, Sampler


class EpochOffsetSampler(Sampler):
    def __init__(self, sampler, epoch, offset):
        if not 0 <= offset <= len(sampler):
            raise ValueError(
                f"resume sample offset {offset} exceeds epoch size {len(sampler)}"
            )
        self.sampler = sampler
        self.resume_epoch = int(epoch)
        self.epoch = int(epoch)
        self.offset = int(offset)
        self.set_epoch(epoch)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    @property
    def active_offset(self):
        return self.offset if self.epoch == self.resume_epoch else 0

    def __len__(self):
        return len(self.sampler) - self.active_offset

    def __iter__(self):
        return islice(iter(self.sampler), self.active_offset, None)


def resume_dataloader(dataloader, epoch, offset, *, drop_last=False):
    """Slice first, then let Accelerate repartition. Pad the final global batch.

    Padding can repeat a few examples, but must never drop unseen tail examples
    merely because the world size increased at a checkpoint boundary.
    """
    kwargs = dict(
        dataset=dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=EpochOffsetSampler(dataloader.sampler, epoch, offset),
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=drop_last,
        worker_init_fn=dataloader.worker_init_fn,
        generator=dataloader.generator,
        timeout=dataloader.timeout,
    )
    if dataloader.num_workers:
        kwargs.update(
            persistent_workers=dataloader.persistent_workers,
            prefetch_factor=dataloader.prefetch_factor,
            multiprocessing_context=dataloader.multiprocessing_context,
        )
    return DataLoader(**kwargs)
