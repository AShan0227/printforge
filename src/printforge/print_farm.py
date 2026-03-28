"""
Print Farm — Direct printer integration for automated printing.

Supports:
  - Bambu Lab (X1C, P1S, A1) via MQTT/cloud API
  - Klipper (Moonraker API)
  - OctoPrint (REST API)

This is the "last mile" — from 3D model to physical object.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class PrinterType(Enum):
    BAMBU = "bambu"
    KLIPPER = "klipper"
    OCTOPRINT = "octoprint"


class PrintStatus(Enum):
    IDLE = "idle"
    PRINTING = "printing"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class Printer:
    """A connected 3D printer."""
    id: str
    name: str
    type: PrinterType
    host: str  # IP or hostname
    port: int = 0
    api_key: str = ""
    serial: str = ""
    model: str = ""  # "X1C", "P1S", "A1", etc.
    status: PrintStatus = PrintStatus.OFFLINE
    # Current job
    current_file: str = ""
    progress: float = 0
    eta_minutes: int = 0
    # Capabilities
    has_ams: bool = False  # Automatic Material System (multi-color)
    has_camera: bool = False
    build_volume: tuple = (256, 256, 256)  # mm


@dataclass
class PrintJob:
    """A print job sent to a printer."""
    id: str
    printer_id: str
    file_path: str
    status: PrintStatus = PrintStatus.IDLE
    progress: float = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None


class PrintFarm:
    """Manage multiple printers and print jobs."""

    def __init__(self):
        self.printers: Dict[str, Printer] = {}
        self.jobs: Dict[str, PrintJob] = {}

    # ── Printer Discovery ────────────────────────────────────────────

    def discover_bambu(self, access_code: str = "", serial: str = "") -> List[Printer]:
        """Discover Bambu Lab printers on local network via SSDP/mDNS."""
        discovered = []

        # Try mDNS discovery
        try:
            import socket
            # Bambu printers advertise on _bambu._tcp
            # Simple approach: scan common ports
            for ip_suffix in range(1, 255):
                ip = f"192.168.1.{ip_suffix}"  # TODO: detect actual subnet
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.1)
                    result = sock.connect_ex((ip, 8883))  # Bambu MQTT port
                    sock.close()
                    if result == 0:
                        printer = Printer(
                            id=f"bambu_{ip}",
                            name=f"Bambu @ {ip}",
                            type=PrinterType.BAMBU,
                            host=ip,
                            port=8883,
                            api_key=access_code,
                            serial=serial,
                        )
                        discovered.append(printer)
                        self.printers[printer.id] = printer
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Bambu discovery failed: {e}")

        return discovered

    def add_printer(self, printer: Printer):
        """Manually add a printer."""
        self.printers[printer.id] = printer
        logger.info(f"Added printer: {printer.name} ({printer.type.value})")

    def add_klipper(self, host: str, port: int = 7125, name: str = "") -> Printer:
        """Add a Klipper printer via Moonraker API."""
        printer = Printer(
            id=f"klipper_{host}",
            name=name or f"Klipper @ {host}",
            type=PrinterType.KLIPPER,
            host=host,
            port=port,
        )
        # Verify connection
        try:
            r = requests.get(f"http://{host}:{port}/printer/info", timeout=5)
            if r.status_code == 200:
                info = r.json().get("result", {})
                printer.status = PrintStatus.IDLE
                printer.model = info.get("software_version", "")
                logger.info(f"Klipper connected: {info}")
        except Exception as e:
            logger.warning(f"Klipper connection failed: {e}")
            printer.status = PrintStatus.OFFLINE

        self.printers[printer.id] = printer
        return printer

    def add_octoprint(self, host: str, api_key: str, port: int = 80, name: str = "") -> Printer:
        """Add an OctoPrint instance."""
        printer = Printer(
            id=f"octoprint_{host}",
            name=name or f"OctoPrint @ {host}",
            type=PrinterType.OCTOPRINT,
            host=host,
            port=port,
            api_key=api_key,
        )
        # Verify
        try:
            r = requests.get(
                f"http://{host}:{port}/api/version",
                headers={"X-Api-Key": api_key},
                timeout=5,
            )
            if r.status_code == 200:
                printer.status = PrintStatus.IDLE
                printer.model = r.json().get("text", "")
        except Exception:
            printer.status = PrintStatus.OFFLINE

        self.printers[printer.id] = printer
        return printer

    # ── Status ───────────────────────────────────────────────────────

    def get_status(self, printer_id: str) -> Optional[Printer]:
        """Get current printer status."""
        printer = self.printers.get(printer_id)
        if not printer:
            return None

        if printer.type == PrinterType.KLIPPER:
            self._update_klipper_status(printer)
        elif printer.type == PrinterType.OCTOPRINT:
            self._update_octoprint_status(printer)

        return printer

    def list_printers(self) -> List[Printer]:
        """List all registered printers."""
        return list(self.printers.values())

    # ── Print ────────────────────────────────────────────────────────

    def send_to_printer(
        self,
        printer_id: str,
        file_path: str,
        auto_start: bool = True,
    ) -> PrintJob:
        """Send a file to a printer and optionally start printing."""
        printer = self.printers.get(printer_id)
        if not printer:
            raise ValueError(f"Unknown printer: {printer_id}")

        job_id = f"job_{int(time.time())}_{printer_id}"
        job = PrintJob(id=job_id, printer_id=printer_id, file_path=file_path)

        if printer.type == PrinterType.KLIPPER:
            self._send_klipper(printer, file_path, auto_start)
        elif printer.type == PrinterType.OCTOPRINT:
            self._send_octoprint(printer, file_path, auto_start)
        elif printer.type == PrinterType.BAMBU:
            self._send_bambu(printer, file_path, auto_start)

        job.status = PrintStatus.PRINTING if auto_start else PrintStatus.IDLE
        job.started_at = time.time() if auto_start else None
        self.jobs[job_id] = job
        return job

    # ── Klipper (Moonraker) ──────────────────────────────────────────

    def _send_klipper(self, printer: Printer, file_path: str, auto_start: bool):
        """Upload and print via Moonraker API."""
        filename = Path(file_path).name

        # Upload
        with open(file_path, 'rb') as f:
            r = requests.post(
                f"http://{printer.host}:{printer.port}/server/files/upload",
                files={"file": (filename, f)},
                timeout=30,
            )
            r.raise_for_status()

        # Start print
        if auto_start:
            requests.post(
                f"http://{printer.host}:{printer.port}/printer/print/start",
                json={"filename": filename},
                timeout=10,
            )

    def _update_klipper_status(self, printer: Printer):
        """Update Klipper printer status."""
        try:
            r = requests.get(
                f"http://{printer.host}:{printer.port}/printer/objects/query?print_stats",
                timeout=5,
            )
            if r.status_code == 200:
                stats = r.json().get("result", {}).get("status", {}).get("print_stats", {})
                state = stats.get("state", "")
                if state == "printing":
                    printer.status = PrintStatus.PRINTING
                    printer.current_file = stats.get("filename", "")
                elif state == "complete":
                    printer.status = PrintStatus.FINISHED
                elif state == "error":
                    printer.status = PrintStatus.ERROR
                else:
                    printer.status = PrintStatus.IDLE
        except Exception:
            printer.status = PrintStatus.OFFLINE

    # ── OctoPrint ────────────────────────────────────────────────────

    def _send_octoprint(self, printer: Printer, file_path: str, auto_start: bool):
        """Upload and print via OctoPrint API."""
        filename = Path(file_path).name
        headers = {"X-Api-Key": printer.api_key}

        with open(file_path, 'rb') as f:
            r = requests.post(
                f"http://{printer.host}:{printer.port}/api/files/local",
                headers=headers,
                files={"file": (filename, f, "application/octet-stream")},
                data={"print": "true" if auto_start else "false"},
                timeout=30,
            )
            r.raise_for_status()

    def _update_octoprint_status(self, printer: Printer):
        """Update OctoPrint status."""
        try:
            r = requests.get(
                f"http://{printer.host}:{printer.port}/api/job",
                headers={"X-Api-Key": printer.api_key},
                timeout=5,
            )
            if r.status_code == 200:
                data = r.json()
                state = data.get("state", "")
                if "Printing" in state:
                    printer.status = PrintStatus.PRINTING
                    printer.progress = data.get("progress", {}).get("completion", 0) or 0
                    printer.eta_minutes = int((data.get("progress", {}).get("printTimeLeft", 0) or 0) / 60)
                elif "Operational" in state:
                    printer.status = PrintStatus.IDLE
                else:
                    printer.status = PrintStatus.OFFLINE
        except Exception:
            printer.status = PrintStatus.OFFLINE

    # ── Bambu Lab ────────────────────────────────────────────────────

    def _send_bambu(self, printer: Printer, file_path: str, auto_start: bool):
        """Send to Bambu Lab printer.

        Bambu uses MQTT over TLS for communication.
        For now, we use the FTP upload method (simpler).
        """
        # Bambu printers accept FTP upload on port 990 (FTPS)
        try:
            import ftplib
            ftp = ftplib.FTP_TLS()
            ftp.connect(printer.host, 990)
            ftp.login("bblp", printer.api_key)  # access_code as password
            ftp.prot_p()

            filename = Path(file_path).name
            with open(file_path, 'rb') as f:
                ftp.storbinary(f"STOR /sdcard/{filename}", f)

            ftp.quit()
            logger.info(f"Uploaded {filename} to Bambu @ {printer.host}")

            # Auto-start requires MQTT command — complex, skip for now
            if auto_start:
                logger.info("Auto-start for Bambu requires MQTT — file uploaded, start manually in Bambu Studio")

        except Exception as e:
            logger.error(f"Bambu upload failed: {e}")
            raise
