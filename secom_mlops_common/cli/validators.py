"""Argparse validators shared by SECOM MLOps scripts."""

import argparse
import math


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite and >= 0")
    return parsed


def positive_int_list(raw_value: str, name: str) -> list[int]:
    values = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError(f"{name} must contain one or more positive integers")
    return values


def probability_list(raw_value: str, name: str) -> list[float]:
    values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values or any(value < 0.0 or value > 1.0 or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain one or more finite values between 0 and 1")
    return values
