from __future__ import annotations


class MiniMLException(Exception):
    """Base exception for the mini-ml framework."""


class StepNotFoundError(MiniMLException):
    pass


class DuplicateStepError(MiniMLException):
    pass


class InvalidSegmentError(MiniMLException):
    pass
