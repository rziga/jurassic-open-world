from abc import ABC
from pathlib import Path
from typing import Any, Generic, Optional, Protocol, TypeVar

import yaml


_ConfigT = TypeVar("_ConfigT", bound="BaseConfig", covariant=True)
_ConfigurableT = TypeVar("_ConfigurableT", bound="BaseConfigurable")


class BaseConfig(ABC, Generic[_ConfigurableT]):
    _target_class: Optional[type[_ConfigurableT]] = None
    _registry: dict[str, type] = {}

    def build(self, *args: Any, **kwargs: Any) -> _ConfigurableT:
        if self._target_class is None:
            raise AttributeError(
                f"Config {self.__class__.__name__} should overwrite `_target_class`."
            )
        return self._target_class(self, *args, **kwargs)

    def __init_subclass__(cls) -> None:
        cls._registry[cls.__name__] = cls

    def to_dict(self) -> dict:
        out_dict = {}
        out_dict["_type"] = self.__class__.__name__
        for k, v in vars(self).items():
            if not isinstance(v, BaseConfig):
                out_dict[k] = v
            else:
                sub_dict = v.to_dict()
                out_dict[k] = sub_dict
        return out_dict

    def to_yaml(self, fpath: str | Path) -> None:
        with open(fpath, "w") as f:
            yaml.safe_dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls: type[_ConfigT], data: dict) -> _ConfigT:
        assert cls.__name__ == data.pop("_type")
        kwargs = {}
        for k, v in data.items():
            if isinstance(v, dict) and "_type" in v:
                subcls = cls._registry[v["_type"]]
                v = subcls.from_dict(v)
            kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls: type[_ConfigT], fpath: str | Path) -> _ConfigT:
        with open(fpath, "r") as f:
            dict = yaml.safe_load(f)
        return cls.from_dict(dict)


class BaseConfigurable(Protocol[_ConfigT]):
    def __init__(self, cfg: _ConfigT, *args: Any, **kwargs: Any): ...
