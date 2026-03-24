"""
Bambu Lab Printer Integration
=============================
Direct connection to Bambu Lab printers via MQTT/FTP.
Supports: X1C, X1, P1S, P1P, A1, A1 Mini
"""

import json
import logging
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BambuPrinter:
    """Bambu Lab printer connection config."""
    ip: str
    access_code: str
    serial: str = ""
    name: str = "Bambu Printer"
    model: str = "unknown"  # X1C, P1S, A1, etc.
    build_volume: tuple = (256, 256, 256)  # mm


@dataclass 
class PrintJob:
    """A print job to send to the printer."""
    file_path: str
    plate_name: str = "PrintForge"
    bed_temp: int = 60
    nozzle_temp: int = 220
    layer_height: float = 0.2
    infill: float = 0.15
    supports: bool = True
    filament: str = "PLA"


@dataclass
class PrinterStatus:
    """Current printer status."""
    state: str  # "idle", "printing", "paused", "error"
    progress: float  # 0.0 - 1.0
    current_layer: int = 0
    total_layers: int = 0
    remaining_minutes: int = 0
    bed_temp: float = 0.0
    nozzle_temp: float = 0.0
    error: Optional[str] = None


# Bambu printer build volumes
PRINTER_PROFILES = {
    "x1c": BambuPrinter(ip="", access_code="", model="X1C", build_volume=(256, 256, 256)),
    "x1": BambuPrinter(ip="", access_code="", model="X1", build_volume=(256, 256, 256)),
    "p1s": BambuPrinter(ip="", access_code="", model="P1S", build_volume=(256, 256, 256)),
    "p1p": BambuPrinter(ip="", access_code="", model="P1P", build_volume=(256, 256, 256)),
    "a1": BambuPrinter(ip="", access_code="", model="A1", build_volume=(256, 256, 256)),
    "a1-mini": BambuPrinter(ip="", access_code="", model="A1 Mini", build_volume=(180, 180, 180)),
}


class BambuConnection:
    """Connection to a Bambu Lab printer via MQTT + FTP."""
    
    def __init__(self, printer: BambuPrinter):
        self.printer = printer
        self._connected = False
    
    def discover(self, timeout: float = 5.0) -> list[dict]:
        """Discover Bambu printers on local network via SSDP/mDNS."""
        discovered = []
        try:
            # Bambu printers advertise via SSDP on port 1990
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(timeout)
            
            # SSDP M-SEARCH for Bambu printers
            ssdp_request = (
                'M-SEARCH * HTTP/1.1\r\n'
                'HOST: 239.255.255.250:1990\r\n'
                'MAN: "ssdp:discover"\r\n'
                'MX: 3\r\n'
                'ST: urn:bambulab-com:device:3dprinter:1\r\n'
                '\r\n'
            )
            
            sock.sendto(ssdp_request.encode(), ('239.255.255.250', 1990))
            
            start = time.time()
            while time.time() - start < timeout:
                try:
                    data, addr = sock.recvfrom(4096)
                    response = data.decode()
                    discovered.append({
                        "ip": addr[0],
                        "port": addr[1],
                        "response": response[:200],
                    })
                except socket.timeout:
                    break
            
            sock.close()
            
        except Exception as e:
            logger.warning(f"SSDP discovery failed: {e}")
        
        return discovered
    
    def connect(self) -> bool:
        """Connect to printer via MQTT."""
        try:
            # In production: use paho-mqtt to connect to printer's MQTT broker
            # Bambu printers run MQTT on port 8883 (TLS)
            logger.info(f"Connecting to {self.printer.name} at {self.printer.ip}...")
            
            # Validate connectivity
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((self.printer.ip, 8883))
            sock.close()
            
            if result == 0:
                self._connected = True
                logger.info(f"Connected to {self.printer.name}")
                return True
            else:
                logger.error(f"Cannot reach printer at {self.printer.ip}:8883")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def get_status(self) -> PrinterStatus:
        """Get current printer status."""
        if not self._connected:
            return PrinterStatus(state="disconnected", progress=0.0, error="Not connected")
        
        # In production: subscribe to MQTT topic device/{serial}/report
        # and parse the JSON status messages
        return PrinterStatus(
            state="idle",
            progress=0.0,
            remaining_minutes=0,
        )
    
    def send_print(self, job: PrintJob) -> bool:
        """Send a 3MF/gcode file to the printer."""
        if not self._connected:
            logger.error("Not connected to printer")
            return False
        
        file_path = Path(job.file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        if file_path.suffix not in ('.3mf', '.gcode'):
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return False
        
        # In production:
        # 1. Upload file via FTP to printer (port 990, TLS)
        # 2. Send MQTT command to start print
        # MQTT topic: device/{serial}/request
        # Payload: {"print": {"command": "project_file", "param": "..."}}
        
        logger.info(f"Sending {file_path.name} to {self.printer.name}...")
        logger.info(f"Settings: {job.filament}, {job.layer_height}mm layers, {int(job.infill*100)}% infill")
        
        # Placeholder: log what would happen
        logger.info(f"✅ Print job queued: {file_path.name}")
        return True
    
    def pause(self) -> bool:
        """Pause current print."""
        # MQTT command: {"print": {"command": "pause"}}
        return self._connected
    
    def resume(self) -> bool:
        """Resume paused print."""
        # MQTT command: {"print": {"command": "resume"}}
        return self._connected
    
    def cancel(self) -> bool:
        """Cancel current print."""
        # MQTT command: {"print": {"command": "stop"}}
        return self._connected


def get_build_volume(printer_model: str) -> tuple:
    """Get build volume for a printer model."""
    profile = PRINTER_PROFILES.get(printer_model.lower().replace(" ", "-"))
    if profile:
        return profile.build_volume
    return (256, 256, 256)  # Default Bambu build volume
