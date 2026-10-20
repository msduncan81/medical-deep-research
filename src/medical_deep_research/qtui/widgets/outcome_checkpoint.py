"""Editable table where the user confirms the outcomes proposed for the GRADE assessment."""

from __future__ import annotations

import json
from typing import Any, Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ...models import ApprovalRequest, ApprovalStatus
from ..theme import SURFACE_SOFT, TEXT_MUTED

OUTCOME_CONFIRMATION_TYPE = "outcome_confirmation"

# column key, i18n key, editor ("text" or a list of choices)
_COLUMNS: list[tuple[str, str, Any]] = [
    ("name", "outcome_col_name", "text"),
    ("importance_score", "outcome_col_importance", [str(n) for n in range(9, 0, -1)]),
    ("kind", "outcome_col_kind", ["binary", "continuous", "time_to_event"]),
    ("desirable_direction", "outcome_col_direction", ["decrease", "increase"]),
    ("target", "outcome_col_target", ["null", "mid"]),
    ("mid_value", "outcome_col_mid", "text"),
    ("mid_unit", "outcome_col_unit", "text"),
    ("mid_source", "outcome_col_source", "text"),
]


def find_outcome_approval(approvals: list[ApprovalRequest]) -> tuple[ApprovalRequest | None, dict[str, Any]]:
    for approval in approvals:
        if approval.status != ApprovalStatus.PENDING.value or not approval.details_json:
            continue
        try:
            details = json.loads(approval.details_json)
        except json.JSONDecodeError:
            continue
        if isinstance(details, dict) and details.get("type") == OUTCOME_CONFIRMATION_TYPE:
            return approval, details
    return None, {}


class OutcomeCheckpointPanel(QFrame):
    """Shows the proposed outcomes; the user edits them and confirms, or skips."""

    confirmed = Signal(str, list)  # noqa: N815  (approval id, outcomes)
    skipped = Signal(str)  # noqa: N815

    def __init__(self, t: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._t = t
        self._approval_id: str | None = None
        self._proposed: list[dict[str, Any]] = []
        self.setObjectName("outcomeCheckpointPanel")
        self.setStyleSheet(
            f"QFrame#outcomeCheckpointPanel {{ background: {SURFACE_SOFT}; border: none; border-radius: 7px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        self._title = QLabel()
        self._title.setStyleSheet("font-weight: 600;")
        self._desc = QLabel()
        self._desc.setWordWrap(True)
        self._desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        layout.addWidget(self._title)
        layout.addWidget(self._desc)
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMinimumHeight(180)
        layout.addWidget(self._table)
        buttons = QHBoxLayout()
        self._add = QPushButton()
        self._remove = QPushButton()
        self._skip = QPushButton()
        self._confirm = QPushButton()
        self._add.clicked.connect(lambda: self._append_row({}))
        self._remove.clicked.connect(self._remove_selected)
        self._skip.clicked.connect(self._on_skip)
        self._confirm.clicked.connect(self._on_confirm)
        for button in (self._add, self._remove):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self._skip)
        buttons.addWidget(self._confirm)
        layout.addLayout(buttons)
        self.retranslate()
        self.setVisible(False)

    def retranslate(self) -> None:
        self._title.setText(self._t("outcome_checkpoint_title"))
        self._desc.setText(self._t("outcome_checkpoint_desc"))
        self._table.setHorizontalHeaderLabels([self._t(key) for _, key, _ in _COLUMNS])
        self._add.setText(self._t("outcome_add"))
        self._remove.setText(self._t("outcome_remove"))
        self._skip.setText(self._t("outcome_skip"))
        self._confirm.setText(self._t("outcome_confirm"))

    def set_approvals(self, approvals: list[ApprovalRequest]) -> None:
        approval, details = find_outcome_approval(approvals)
        if approval is None:
            self._approval_id = None
            self.setVisible(False)
            return
        if approval.id == self._approval_id:
            return  # keep the user's edits while the checkpoint is open
        self._approval_id = approval.id
        self._proposed = [o for o in details.get("proposed") or [] if isinstance(o, dict)]
        self._table.setRowCount(0)
        for outcome in self._proposed:
            self._append_row(outcome)
        self.setVisible(True)

    def _append_row(self, outcome: dict[str, Any]) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        for col, (key, _label, editor) in enumerate(_COLUMNS):
            value = outcome.get(key)
            text = "" if value is None else str(value)
            if isinstance(editor, list):
                combo = QComboBox()
                combo.addItems(editor)
                if text in editor:
                    combo.setCurrentText(text)
                self._table.setCellWidget(row, col, combo)
            else:
                self._table.setItem(row, col, QTableWidgetItem(text))
        # keep the id and rationale of proposed outcomes so edits stay linked to them
        first = self._table.item(row, 0)
        if first is not None:
            first.setData(256, {"id": outcome.get("id"), "rationale": outcome.get("rationale", "")})

    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self._table.selectedIndexes()}, reverse=True)
        for row in rows:
            self._table.removeRow(row)

    def table_outcomes(self) -> list[dict[str, Any]]:
        outcomes: list[dict[str, Any]] = []
        for row in range(self._table.rowCount()):
            data: dict[str, Any] = {}
            for col, (key, _label, editor) in enumerate(_COLUMNS):
                if isinstance(editor, list):
                    widget = self._table.cellWidget(row, col)
                    data[key] = widget.currentText() if isinstance(widget, QComboBox) else editor[0]
                else:
                    item = self._table.item(row, col)
                    data[key] = item.text().strip() if item else ""
            if not data["name"]:
                continue
            meta = self._table.item(row, 0).data(256) if self._table.item(row, 0) else None
            meta = meta if isinstance(meta, dict) else {}
            data["id"] = meta.get("id") or f"o{row + 1}"
            data["rationale"] = meta.get("rationale") or ""
            data["importance_score"] = int(data["importance_score"])
            try:
                data["mid_value"] = float(str(data["mid_value"]).replace(",", ".")) if data["mid_value"] else None
            except ValueError:
                data["mid_value"] = None
            for key in ("mid_unit", "mid_source"):
                data[key] = data[key] or None
            outcomes.append(data)
        return outcomes

    def _on_confirm(self) -> None:
        if self._approval_id:
            self.confirmed.emit(self._approval_id, self.table_outcomes())
            self.setVisible(False)

    def _on_skip(self) -> None:
        if self._approval_id:
            self.skipped.emit(self._approval_id)
            self.setVisible(False)
