"""
sliceview — Zero-copy slice views for Python sequences.

A sliceview presents a live window into an existing sequence:
reads and writes reflect the underlying sequence, view-to-view
slicing composes in O(1), and no data is copied unless explicitly
requested.

Basic usage::

    from sliceview import sliceview

    big = list(range(1_000_000))

    # O(1) window — no copy
    window = sliceview(big)[1000:2000]

    # Compose slices — still O(1)
    every_other = sliceview(big)[::2][500:1000]

    # In-place windowed update
    sv = sliceview(big)
    sv[0:10] = range(10, 20)

    # Sliding window without creating new objects
    view = sliceview(big, 0, 100)
    for _ in range(10):
        view.advance(100)
"""

from __future__ import annotations

import reprlib
from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableSequence,
    Sequence,
)
from typing import (
     Any,
    Optional,
    SupportsIndex,
    TypeVar,
    Union,
    overload,
)

__all__ = ["sliceview"]
__version__ = "0.1.1"

T = TypeVar("T")


class _OpenRange:
    """Sentinel for a range whose stop is open (None), so it grows with the base."""

    __slots__ = ("start", "step", "open_start")

    def __init__(self, start: int, step: int, open_start: bool = False) -> None:
        self.start = start
        self.step = step
        self.open_start = open_start

    def resolve(self, b_len: int) -> range:
        if self.step > 0:
            return range(min(self.start, b_len), b_len, self.step)
        if b_len == 0:
            return range(0, 0, self.step)
        start = b_len - 1 if self.open_start else min(self.start, b_len - 1)
        return range(start, -1, self.step)


class sliceview(Sequence[T]):
    """A zero-copy, composable slice view over any :class:`collections.abc.Sequence`.

    Parameters
    ----------
    base:
        The underlying sequence.  Any object that implements
        ``__len__`` and ``__getitem__`` with integer indices is accepted
        (mappings such as :class:`dict` are rejected).
    start, stop, step:
        Slice parameters (same semantics as the built-in :class:`slice`).
        *start* may alternatively be a :class:`slice` object, in which case
        *stop* and *step* must be omitted.

    Examples
    --------
    >>> sv = sliceview([0, 1, 2, 3, 4, 5])
    >>> list(sv[1:4])
    [1, 2, 3]
    >>> list(sv[::2])
    [0, 2, 4]
    >>> sv2 = sv[1:][::2]   # composed — O(1), no copy
    >>> list(sv2)
    [1, 3, 5]
    """

    __slots__ = ("_base", "_range")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @overload
    def __init__(self, base: Sequence[T]) -> None: ...
    @overload
    def __init__(self, base: Sequence[T], start: slice) -> None: ...
    @overload
    def __init__(
        self,
        base: Sequence[T],
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None: ...

    def __init__(
        self,
        base: Sequence[T],
        start: Union[int, slice, None] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        if isinstance(base, Mapping) or not (
            hasattr(base, "__len__") and hasattr(base, "__getitem__")
        ):
            raise TypeError(
                f"sliceview requires a sequence with __len__ and __getitem__, "
                f"got {type(base).__name__!r}"
            )

        if isinstance(start, slice):
            if stop is not None or step is not None:
                raise ValueError(
                    "sliceview initialized with slice "
                    "must not have stop/step arguments"
                )
            sl: slice = start
        else:
            sl = slice(start, stop, step)

        # Unwrap nested sliceview so .base always points to the original sequence.
        if isinstance(base, sliceview):
            self._base: Union[Sequence[T], MutableSequence[T]] = base._base
            self._range: Union[range, _OpenRange] = base._current_range()[sl]
            return

        self._base = base
        b_len = len(base)
        s, e, st = sl.indices(b_len)

        # If the original stop was None, store an open sentinel so that the
        # view grows when elements are appended to the base.
        if sl.stop is None:
            self._range = _OpenRange(s, st, open_start=(sl.start is None and st < 0))
        else:
            self._range = range(s, e, st)

    @classmethod
    def _from_range(cls, base: Sequence[T], r: Union[range, _OpenRange]) -> sliceview[T]:
        """Internal constructor: build a view directly from a range object."""
        sv = cls.__new__(cls)
        sv._base = base
        sv._range = r
        return sv

    # ------------------------------------------------------------------
    # Core helper
    # ------------------------------------------------------------------

    def _current_range(self) -> range:
        """Return a concrete ``range`` clamped to the current base length."""
        r = self._range
        b_len = len(self._base)
        if isinstance(r, _OpenRange):
            return r.resolve(b_len)
        if r.step > 0:
            if r.stop <= b_len:
                return r
            return range(min(r.start, b_len), min(r.stop, b_len), r.step)
        else:
            if r.start < b_len:
                return r
            if b_len == 0:
                return range(0, 0, r.step)
            skip = (r.start - (b_len - 1) + (-r.step) - 1) // (-r.step)
            return r[skip:]

    # ------------------------------------------------------------------
    # Slice properties
    # ------------------------------------------------------------------

    @property
    def start(self) -> int:
        """Current start index in the underlying sequence."""
        return self._current_range().start

    @property
    def stop(self) -> int:
        """Current stop index in the underlying sequence."""
        return self._current_range().stop

    @property
    def step(self) -> int:
        """Step stride in the underlying sequence."""
        return self._current_range().step

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._current_range())

    @overload
    def __getitem__(self, index: SupportsIndex) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> sliceview[T]: ...

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            # Compose slices using Python's range slicing — O(1), exact.
            sub = self._current_range()[index]
            return sliceview._from_range(self._base, sub)

        if not isinstance(index, SupportsIndex):
            raise TypeError(
                f"{type(self).__name__} indices must be integers or slices, "
                f"not {type(index).__name__}"
            )

        cr = self._current_range()
        try:
            base_idx = cr[index]
        except IndexError:
            raise IndexError("sliceview index out of range") from None
        return self._base[base_idx]

    @overload
    def __setitem__(self, index: SupportsIndex, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...

    def __setitem__(self, index: Any, value: Any) -> None:
        if not hasattr(self._base, "__setitem__"):
            raise TypeError("underlying sequence is not mutable")

        if isinstance(index, slice):
            self._setslice(index, value)
            return

        if not isinstance(index, SupportsIndex):
            raise TypeError(
                f"{type(self).__name__} indices must be integers or slices, "
                f"not {type(index).__name__}"
            )

        cr = self._current_range()
        try:
            base_idx = cr[index]
        except IndexError:
            raise IndexError("sliceview index out of range") from None
        self._base[base_idx] = value  # type: ignore[index]

    def __iter__(self) -> Iterator[T]:
        base = self._base
        for i in self._current_range():
            yield base[i]

    def __contains__(self, item: object) -> bool:
        return any(el is item or el == item for el in self)

    def __reversed__(self) -> Iterator[T]:
        base = self._base
        return (base[i] for i in reversed(self._current_range()))

    # ------------------------------------------------------------------
    # Equality and hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, Sequence):
            return len(self) == len(other) and all(a == b for a, b in zip(self, other))
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]  # unhashable by design

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cr = self._current_range()
        return f"sliceview({reprlib.repr(self._base)})[{cr.start}:{cr.stop}:{cr.step}]"

    # ------------------------------------------------------------------
    # Advance — sliding-window helper
    # ------------------------------------------------------------------

    def advance(self, n: int) -> sliceview[T]:
        """Shift the view's window forward by *n* index positions in-place.

        Returns *self* so calls can be chained.  Useful for sliding windows:

        >>> data = list(range(10))
        >>> sv = sliceview(data, 0, 3)
        >>> list(sv)
        [0, 1, 2]
        >>> sv.advance(3)                    # doctest: +ELLIPSIS
        sliceview(...)[3:6:1]
        >>> list(sv)
        [3, 4, 5]

        Parameters
        ----------
        n:
            Positions to advance (negative to retreat).
        """
        b_len = len(self._base)
        cr = self._current_range()
        if cr.step > 0:
            new_start = max(0, min(cr.start + n, b_len))
            delta = new_start - cr.start
            new_stop = max(0, min(cr.stop + delta, b_len))
        else:
            new_start = max(-1, min(cr.start + n, b_len - 1))
            delta = new_start - cr.start
            new_stop = max(-1, min(cr.stop + delta, b_len - 1))
        self._range = range(new_start, new_stop, cr.step)
        return self

    # ------------------------------------------------------------------
    # tolist / copy
    # ------------------------------------------------------------------

    def tolist(self) -> list[T]:
        """Return a new list containing the elements covered by this view."""
        return list(self)

    def copy(self) -> list[T]:
        """Alias for :meth:`tolist`."""
        return self.tolist()

    # ------------------------------------------------------------------
    # base property
    # ------------------------------------------------------------------

    @property
    def base(self) -> Union[Sequence[T], MutableSequence[T]]:
        """The underlying sequence this view points into."""
        return self._base

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setslice(self, sl: slice, values: Iterable[T]) -> None:
        target_range = self._current_range()[sl]
        values_list = list(values)

        if target_range.step != 1:
            # Extended slice assignment (including step == -1) must match length exactly.
            if len(values_list) != len(target_range):
                raise ValueError(
                    f"attempt to assign sequence of size {len(values_list)} "
                    f"to extended slice of size {len(target_range)}"
                )
            for i, v in zip(target_range, values_list):
                self._base[i] = v  # type: ignore[index]
        else:
            # Step 1: delegate to the base (allows resizing if base supports it).
            self._base[  # type: ignore[index]
                slice(target_range.start, target_range.stop, target_range.step)
            ] = values_list
