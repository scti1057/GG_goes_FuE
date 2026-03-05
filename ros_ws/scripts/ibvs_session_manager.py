#!/usr/bin/env python3
"""Interactive IBVS session manager.

Run this script in a sourced ROS2 shell, e.g.:
  source /home/ros_ws/install/setup.bash
  python3 /home/ros_ws/scripts/ibvs_session_manager.py --debug
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ManagedProcess:
    name: str
    cmd: List[str]
    log_path: Path
    process: Optional[subprocess.Popen] = None
    _log_handle: Optional[object] = field(default=None, repr=False)

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> bool:
        if self.is_running():
            return False

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        self._log_handle.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {self.name}\n"
            f"CMD: {' '.join(shlex.quote(x) for x in self.cmd)}\n\n"
        )
        self._log_handle.flush()

        self.process = subprocess.Popen(
            self.cmd,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
        )
        return True

    def stop(self, timeout_sec: float = 5.0) -> bool:
        if self.process is None:
            return False

        was_running = self.is_running()
        if was_running:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
            except ProcessLookupError:
                pass

            try:
                self.process.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=2.0)

        if self._log_handle is not None:
            self._log_handle.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] STOP {self.name}\n"
            )
            self._log_handle.close()
            self._log_handle = None

        self.process = None
        return was_running


def run_cmd(cmd: List[str], timeout_sec: float = 8.0) -> Tuple[int, str]:
    try:
        cp = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        out = (cp.stdout or "") + (cp.stderr or "")
        return cp.returncode, out.strip()
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        if out.strip():
            return 124, out.strip()
        return 124, f"Timeout after {timeout_sec:.1f}s: {' '.join(cmd)}"


class IbvsSessionManager:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.log_dir = Path(args.log_dir)
        self.processes = self._build_processes()

    def _build_processes(self) -> dict:
        keypoint_cmd = [
            "ros2",
            "run",
            "ibvs_perception",
            "keypoint",
            "--ros-args",
            "-p",
            f"detector_type:={self.args.detector_type}",
            "-p",
            f"device:={self.args.device}",
            "-p",
            f"top_k:={self.args.top_k}",
            "-p",
            f"debug_mode:={'true' if self.args.debug else 'false'}",
            "-p",
            f"input_topic:={self.args.input_topic}",
            "-p",
            f"depth_topic:={self.args.depth_topic}",
            "-p",
            f"output_topic:={self.args.keypoint_debug_topic}",
            "-p",
            f"binary_output_topic:={self.args.mask_debug_topic}",
        ]
        if self.args.xfeat_repo_dir:
            keypoint_cmd += ["-p", f"xfeat_repo_dir:={self.args.xfeat_repo_dir}"]

        reference_cmd = [
            "ros2",
            "run",
            "ibvs_reference",
            "reference_manager",
            "--ros-args",
            "-p",
            f"keypoints_topic:={self.args.keypoints_topic}",
            "-p",
            f"image_topic:={self.args.input_topic}",
            "-p",
            f"init_duration_sec:={self.args.init_duration_sec}",
            "-p",
            f"ref_top_k:={self.args.ref_top_k}",
            "-p",
            f"debug_mode:={'true' if self.args.debug else 'false'}",
            "-p",
            f"debug_topic:={self.args.reference_debug_topic}",
        ]

        matcher_cmd = [
            "ros2",
            "run",
            "ibvs_matching",
            "descriptor_matcher",
            "--ros-args",
            "-p",
            f"keypoints_topic:={self.args.keypoints_topic}",
            "-p",
            f"reference_topic:={self.args.reference_topic}",
            "-p",
            f"matches_topic:={self.args.matches_topic}",
            "-p",
            f"match_threshold:={self.args.match_threshold}",
            "-p",
            f"mutual_check:={'true' if self.args.mutual_check else 'false'}",
        ]

        matches_viz_cmd = [
            "ros2",
            "run",
            "ibvs_matching",
            "matches_viz",
            "--ros-args",
            "-p",
            f"image_topic:={self.args.input_topic}",
            "-p",
            f"matches_topic:={self.args.matches_topic}",
            "-p",
            f"output_topic:={self.args.matches_debug_topic}",
        ]

        return {
            "keypoint": ManagedProcess("keypoint", keypoint_cmd, self.log_dir / "keypoint.log"),
            "reference_manager": ManagedProcess(
                "reference_manager", reference_cmd, self.log_dir / "reference_manager.log"
            ),
            "descriptor_matcher": ManagedProcess(
                "descriptor_matcher", matcher_cmd, self.log_dir / "descriptor_matcher.log"
            ),
            "matches_viz": ManagedProcess("matches_viz", matches_viz_cmd, self.log_dir / "matches_viz.log"),
        }

    def _is_core_running(self) -> bool:
        return self.processes["keypoint"].is_running() and self.processes["reference_manager"].is_running()

    def _is_tracking_running(self) -> bool:
        return self.processes["descriptor_matcher"].is_running() and self.processes["matches_viz"].is_running()

    def start_core(self) -> None:
        print("\n[core] Starte keypoint + reference_manager ...")
        for name in ("keypoint", "reference_manager"):
            proc = self.processes[name]
            started = proc.start()
            if started:
                print(f"  - {name} gestartet (pid={proc.process.pid})")
            else:
                print(f"  - {name} läuft bereits (pid={proc.process.pid})")
        print(f"[core] Logs: {self.log_dir}")

    def stop_core(self) -> None:
        print("\n[core] Stoppe keypoint + reference_manager ...")
        for name in ("reference_manager", "keypoint"):
            stopped = self.processes[name].stop()
            print(f"  - {name}: {'gestoppt' if stopped else 'war bereits aus'}")

    def start_tracking(self) -> None:
        init_done = self.read_init_done()
        if init_done is not True:
            print("\n[tracking] Nicht erlaubt: Initialisierung ist noch nicht abgeschlossen.")
            return

        if not self._is_core_running():
            print("\n[tracking] Core läuft nicht, starte zuerst core nodes.")
            self.start_core()

        print("\n[tracking] Starte descriptor_matcher + matches_viz ...")
        for name in ("descriptor_matcher", "matches_viz"):
            proc = self.processes[name]
            started = proc.start()
            if started:
                print(f"  - {name} gestartet (pid={proc.process.pid})")
            else:
                print(f"  - {name} läuft bereits (pid={proc.process.pid})")

    def stop_tracking(self) -> None:
        print("\n[tracking] Stoppe descriptor_matcher + matches_viz ...")
        for name in ("matches_viz", "descriptor_matcher"):
            stopped = self.processes[name].stop()
            print(f"  - {name}: {'gestoppt' if stopped else 'war bereits aus'}")

    def start_initialization(self) -> None:
        if self._is_tracking_running():
            print("\n[init] Tracking läuft. Stoppe Tracking zuerst ...")
            self.stop_tracking()

        if not self._is_core_running():
            print("\n[init] Core läuft nicht, starte core nodes.")
            self.start_core()
            time.sleep(1.0)

        ok, message, raw = self.call_start_capture_service()
        if not ok:
            print("\n[init] Konnte Initialisierung nicht starten.")
            print(raw)
            return

        print(f"\n[init] {message}")
        print("[init] Warte auf /ibvs/init_done=true ...")
        done = self.wait_for_init_done(timeout_sec=self.args.init_wait_timeout)
        if done:
            print("[init] Initialisierung abgeschlossen.")
        else:
            print("[init] Timeout beim Warten auf init_done=true.")

    def stop_all(self) -> None:
        self.stop_tracking()
        self.stop_core()

    def call_start_capture_service(self) -> Tuple[bool, str, str]:
        cmd = [
            "ros2",
            "service",
            "call",
            "/ibvs/reference/start_capture",
            "std_srvs/srv/Trigger",
            "{}",
        ]
        rc, out = run_cmd(cmd, timeout_sec=12.0)
        if rc != 0:
            return False, "Service-Call fehlgeschlagen.", out

        success = "success=True" in out
        match = re.search(r"message='([^']*)'", out)
        message = match.group(1) if match else "(keine message)"
        return success, message, out

    def read_init_done(self) -> Optional[bool]:
        cmd = [
            "ros2",
            "topic",
            "echo",
            "/ibvs/init_done",
            "--once",
            "--qos-durability",
            "transient_local",
            "--qos-reliability",
            "reliable",
            "--qos-history",
            "keep_last",
            "--qos-depth",
            "1",
        ]
        rc, out = run_cmd(cmd, timeout_sec=6.0)
        out_low = out.lower()
        if "data: true" in out_low:
            return True
        if "data: false" in out_low:
            return False
        if rc != 0:
            return None
        return None

    def wait_for_init_done(self, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            state = self.read_init_done()
            if state is True:
                return True
            time.sleep(0.5)
        return False

    def print_status(self) -> None:
        init_done = self.read_init_done()
        init_str = "true" if init_done is True else ("false" if init_done is False else "unknown")

        print("\n========== IBVS Status ==========")
        print(f"core_running:     {self._is_core_running()}")
        print(f"tracking_running: {self._is_tracking_running()}")
        print(f"init_done_topic:  {init_str}")
        print(f"log_dir:          {self.log_dir}")
        for name, proc in self.processes.items():
            if proc.is_running():
                print(f"  - {name}: running (pid={proc.process.pid})")
            else:
                print(f"  - {name}: stopped")
        print("=================================\n")

    def print_menu(self) -> None:
        print("Aktionen:")
        print("  1) Core starten (keypoint + reference_manager)")
        print("  2) Initialisierung starten")
        print("  3) Tracking starten (nur wenn init_done=true)")
        print("  4) Tracking stoppen")
        print("  5) Status anzeigen")
        print("  6) Alle Nodes stoppen")
        print("  q) Beenden (stoppt ebenfalls alle Nodes)")

    def run(self) -> int:
        if shutil.which("ros2") is None:
            print("Fehler: 'ros2' wurde nicht gefunden. Bitte zuerst ROS sourcen.")
            return 2

        print("\nIBVS Session Manager gestartet.")
        print("Hinweis: Starte in einem zweiten Terminal bei Bedarf RViz2.")
        print("Hinweis: Node-Logs landen in:", self.log_dir)
        self.print_status()

        try:
            while True:
                self.print_menu()
                choice = input("\nAuswahl: ").strip().lower()

                if choice == "1":
                    self.start_core()
                elif choice == "2":
                    self.start_initialization()
                elif choice == "3":
                    self.start_tracking()
                elif choice == "4":
                    self.stop_tracking()
                elif choice == "5":
                    self.print_status()
                elif choice == "6":
                    self.stop_all()
                elif choice == "q":
                    print("\nBeende Manager und stoppe alle Nodes ...")
                    self.stop_all()
                    return 0
                else:
                    print("Unbekannte Eingabe.")
        except KeyboardInterrupt:
            print("\n\nCtrl+C erkannt. Stoppe alle Nodes ...")
            self.stop_all()
            return 130


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive IBVS startup/tracking manager.")
    parser.add_argument("--debug", action="store_true", help="Enable debug topics for managed nodes.")
    parser.add_argument("--log-dir", default="/tmp/ibvs_manager_logs", help="Directory for node logs.")

    parser.add_argument("--detector-type", default="xfeat")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--xfeat-repo-dir", default="", help="Optional override for xfeat repo path.")

    parser.add_argument("--input-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--keypoints-topic", default="/ibvs/keypoints")
    parser.add_argument("--reference-topic", default="/ibvs/reference/keypoints")
    parser.add_argument("--matches-topic", default="/ibvs/matches")

    parser.add_argument("--keypoint-debug-topic", default="/ibvs/debug/keypoints_image")
    parser.add_argument("--mask-debug-topic", default="/ibvs/debug/near_mask")
    parser.add_argument("--reference-debug-topic", default="/ibvs/debug/reference_candidates_image")
    parser.add_argument("--matches-debug-topic", default="/ibvs/debug/matches_image")

    parser.add_argument("--init-duration-sec", type=float, default=5.0)
    parser.add_argument("--ref-top-k", type=int, default=300)
    parser.add_argument("--init-wait-timeout", type=float, default=20.0)

    parser.add_argument("--match-threshold", type=float, default=0.85)
    parser.add_argument("--mutual-check", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manager = IbvsSessionManager(args)
    return manager.run()


if __name__ == "__main__":
    sys.exit(main())
