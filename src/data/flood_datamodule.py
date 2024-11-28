from typing import Any, Dict, Optional, Tuple

import torch
import rootutils
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.datasets import init_dataset


class FloodDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        dataset_name: str = "flood",
        metadata_path: str = "weights/metadata.json",
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        length: int = 28,
        step: int = 4,
        handle_missing_values: str = "drop",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:

            self.data_train = init_dataset(self.hparams.dataset_name,
                                            data_dir=self.hparams.data_dir,
                                            length=self.hparams.length,
                                            step=self.hparams.step,
                                            metadata_path=self.hparams.metadata_path,
                                            mode="train",
                                            handle_missing_values=self.hparams.handle_missing_values)

            self.data_val = init_dataset(self.hparams.dataset_name,
                                        data_dir=self.hparams.data_dir,
                                        length=self.hparams.length,
                                        step=self.hparams.step,
                                        mode="val",
                                        metadata_path=self.hparams.metadata_path,
                                        handle_missing_values=self.hparams.handle_missing_values)

            self.data_test = init_dataset(self.hparams.dataset_name,
                                        data_dir=self.hparams.data_dir,
                                        length=self.hparams.length,
                                        step=self.hparams.step,
                                        mode="test",
                                        metadata_path=self.hparams.metadata_path,
                                        handle_missing_values=self.hparams.handle_missing_values)

            print('Train-Val-Test:', len(self.data_train), len(self.data_val), len(self.data_test))

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = rootutils.find_root(search_from=__file__,
                                indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "data")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="flood.yaml")
    def main(cfg: DictConfig):
        datamodule: FloodDataModule = hydra.utils.instantiate(cfg, data_dir=f"{root}/data")
        datamodule.setup()
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        test_dataloader = datamodule.val_dataloader()
        print("Length of train dataloader:", len(train_dataloader))
        print("Length of val dataloader:", len(val_dataloader))
        print("Length of test dataloader:", len(test_dataloader))

        input, cond, output = next(iter(train_dataloader))

        print(input.shape, cond.shape, output.shape)

    main()