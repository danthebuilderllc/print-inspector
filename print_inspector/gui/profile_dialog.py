"""The inspection profile library: load, save the current setup, delete, import and export.

Profiles are local JSON files (core/profiles.py); import and export copy them
to or from any local or network folder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from print_inspector.core.profiles import (
    PROFILE_SUFFIX,
    InspectionProfile,
    ProfileEntry,
    ProfileError,
    delete_profile,
    file_stem,
    find_profile,
    list_profiles,
    load_profile,
    path_for,
    save_profile,
)

COLUMNS = ("Name", "Artwork", "Contents", "Modified")
# Builds a profile of the window's current setup: (name, description, profile it replaces or None).
ProfileFactory = Callable[[str, str, "InspectionProfile | None"], InspectionProfile]


class ProfileDialog(QDialog):
    """After ``exec()``: ``chosen()`` is the profile to load (Load) and ``saved()`` the one saved last."""

    def __init__(
        self,
        directory: Path,
        make_profile: ProfileFactory,
        active_name: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Inspection Profiles")
        self._directory = Path(directory)
        self._make_profile = make_profile
        self._active_name = active_name
        self._entries: list[ProfileEntry] = []
        self._chosen: tuple[Path, InspectionProfile] | None = None
        self._saved: tuple[Path, InspectionProfile] | None = None
        # Replaceable in tests: ask a yes/no question.
        self.confirm: Callable[[str, str], bool] = self._ask

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.table.itemDoubleClicked.connect(lambda _: self.load_selected())
        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail_label.setObjectName("sideNote")

        self.load_button = QPushButton("Load")
        self.load_button.setToolTip("Use this profile's settings, regions and zones for the inspection.")
        self.load_button.clicked.connect(self.load_selected)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_selected)
        self.import_button = QPushButton("Import…")
        self.import_button.setToolTip(
            "Copy a profile file (e.g. from a colleague or a network folder) into the library."
        )
        self.import_button.clicked.connect(self._choose_import)
        self.export_button = QPushButton("Export…")
        self.export_button.setToolTip("Save a copy of the selected profile to a file of your choice.")
        self.export_button.clicked.connect(self._choose_export)
        row = QHBoxLayout()
        for button in (self.load_button, self.delete_button, self.import_button, self.export_button):
            row.addWidget(button)
        row.addStretch(1)

        save_box = QGroupBox("Save the current setup as a profile")
        self.name_edit = QLineEdit(active_name)
        self.name_edit.setPlaceholderText("e.g. Product 4711 front label")
        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("optional")
        self.save_button = QPushButton("Save Profile")
        self.save_button.setToolTip(
            "Store the current sensitivity, acceptance criteria, text, barcode and colour settings, regions and "
            "zones under this name (an existing profile of the same name is replaced)."
        )
        self.save_button.clicked.connect(lambda: self.save_current(self.name_edit.text(), self.description_edit.text()))
        form = QFormLayout(save_box)
        form.addRow("Name", self.name_edit)
        form.addRow("Description", self.description_edit)
        form.addRow("", self.save_button)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b91c1c;")
        where = QLabel(f"Profiles are stored on this computer in {self._directory}")
        where.setWordWrap(True)
        where.setObjectName("sideNote")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.detail_label)
        layout.addLayout(row)
        layout.addWidget(save_box)
        layout.addWidget(self.error_label)
        layout.addWidget(where)
        layout.addWidget(buttons)
        width = self.fontMetrics().averageCharWidth() * 110
        self.resize(width, int(width * 0.7))
        self.refresh()

    # ---------------------------------------------------------------- list

    def refresh(self, select: str | None = None) -> None:
        self._entries = list_profiles(self._directory)
        self.table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            p = entry.profile
            name = entry.name + ("  (in use)" if p is not None and p.name == self._active_name else "")
            values = (name, p.artwork_name if p else "", p.summary() if p else entry.error,
                      p.modified.replace("T", " ")[:16] if p else "")
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(entry.error or (p.description if p else "") or str(entry.path))
                if p is None:
                    item.setForeground(QBrush(QColor("#b91c1c")))
                self.table.setItem(row, col, item)
        self.table.resizeColumnToContents(0)
        target = select or self._active_name
        index = next((i for i, e in enumerate(self._entries) if e.name == target), 0 if self._entries else -1)
        if index >= 0:
            self.table.selectRow(index)
        self._on_selection()

    def entries(self) -> list[ProfileEntry]:
        return list(self._entries)

    def selected_entry(self) -> ProfileEntry | None:
        row = self.table.currentRow()
        return self._entries[row] if 0 <= row < len(self._entries) else None

    def select_name(self, name: str) -> bool:
        for row, entry in enumerate(self._entries):
            if entry.name == name:
                self.table.selectRow(row)
                return True
        return False

    def _on_selection(self) -> None:
        entry = self.selected_entry()
        usable = entry is not None and entry.profile is not None
        self.load_button.setEnabled(usable)
        self.export_button.setEnabled(usable)
        self.delete_button.setEnabled(entry is not None)
        if entry is None:
            self.detail_label.setText("No profiles yet. Set up the inspection for a label design and save it below.")
        elif entry.profile is None:
            self.detail_label.setText(f"Cannot be used: {entry.error}\n{entry.path}")
        else:
            p = entry.profile
            lines = [f"<b>{p.name}</b>" + (f" – {p.description}" if p.description else ""), p.summary()]
            if p.artwork_name:
                size = f" ({p.artwork_size[0]} × {p.artwork_size[1]} px)" if p.artwork_size else ""
                lines.append(f"Made with artwork {p.artwork_name}{size}")
            lines.append(f"Created {p.created.replace('T', ' ')[:16]}, changed {p.modified.replace('T', ' ')[:16]}"
                         + (f", Print Inspector {p.app_version}" if p.app_version else ""))
            lines.append(str(entry.path))
            self.detail_label.setText("<br>".join(lines))
            self.name_edit.setText(p.name)
            self.description_edit.setText(p.description)

    # ---------------------------------------------------------------- actions

    def _ask(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(self, title, text)
        return answer == QMessageBox.StandardButton.Yes

    def _error(self, message: str) -> None:
        self.error_label.setText(message)

    def load_selected(self) -> bool:
        entry = self.selected_entry()
        if entry is None or entry.profile is None:
            return False
        self._chosen = (entry.path, entry.profile)
        self.accept()
        return True

    def save_current(self, name: str, description: str = "") -> bool:
        """Save the window's current setup under ``name`` (asks before replacing another profile)."""
        name = name.strip()
        if not name:
            self._error("Enter a name for the profile.")
            return False
        existing = find_profile(self._directory, name)
        if existing is not None and name != self._active_name and not self.confirm(
            "Replace Profile", f"A profile named '{name}' exists. Replace it with the current setup?"
        ):
            return False
        try:
            profile = self._make_profile(name, description, existing.profile if existing else None)
            path = path_for(self._directory, name)
            save_profile(path, profile)
        except (ProfileError, OSError) as exc:
            self._error(f"The profile could not be saved: {exc}")
            return False
        self._saved = (path, profile)
        self._active_name = name
        self._error("")
        self.refresh(select=name)
        return True

    def delete_selected(self) -> bool:
        entry = self.selected_entry()
        if entry is None or not self.confirm(
            "Delete Profile", f"Delete the profile '{entry.name}'?\n{entry.path}\n\nThis cannot be undone."
        ):
            return False
        try:
            delete_profile(entry.path)
        except OSError as exc:
            self._error(f"The profile could not be deleted: {exc}")
            return False
        self.refresh()
        return True

    def import_file(self, path: Path) -> bool:
        """Copy a profile file into the library (validated first)."""
        try:
            profile = load_profile(Path(path))
        except ProfileError as exc:
            self._error(str(exc))
            return False
        if find_profile(self._directory, profile.name) is not None and not self.confirm(
            "Replace Profile", f"A profile named '{profile.name}' exists. Replace it with the imported one?"
        ):
            return False
        try:
            save_profile(path_for(self._directory, profile.name), profile)
        except OSError as exc:
            self._error(f"The profile could not be imported: {exc}")
            return False
        self._error("")
        self.refresh(select=profile.name)
        return True

    def export_selected(self, path: Path) -> bool:
        entry = self.selected_entry()
        if entry is None or entry.profile is None:
            return False
        try:
            save_profile(Path(path), entry.profile)
        except OSError as exc:
            self._error(f"The profile could not be exported: {exc}")
            return False
        self._error("")
        return True

    def _choose_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Profile", str(Path.home()), f"Print Inspector profiles (*{PROFILE_SUFFIX});;All files (*)"
        )
        if path:
            self.import_file(Path(path))

    def _choose_export(self) -> None:
        entry = self.selected_entry()
        if entry is None or entry.profile is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Profile", str(Path.home() / f"{file_stem(entry.profile.name)}{PROFILE_SUFFIX}"),
            f"Print Inspector profiles (*{PROFILE_SUFFIX})",
        )
        if path:
            self.export_selected(Path(path))

    # ---------------------------------------------------------------- results

    def chosen(self) -> tuple[Path, InspectionProfile] | None:
        return self._chosen

    def saved(self) -> tuple[Path, InspectionProfile] | None:
        return self._saved
