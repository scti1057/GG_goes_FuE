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

    @staticmethod
    def _parse_ros_bool_param_get(output: str) -> Optional[bool]:
        out = output.strip().lower()
        # Typical format: "Boolean value is: True"
        m = re.search(r"boolean value is:\s*(true|false)", out)
        if m:
            return m.group(1) == "true"
        # Fallback: plain true/false in output
        if re.search(r"\btrue\b", out):
            return True
        if re.search(r"\bfalse\b", out):
            return False
        return None

    def _set_node_param_bool(
        self,
        node_name: str,
        param_name: str,
        target_value: bool,
        retries: int,
        timeout: float,
        retry_wait: float,
    ) -> bool:
        value_str = "true" if target_value else "false"
        retries = max(1, int(retries))

        for attempt in range(1, retries + 1):
            cmd_set = [
                "ros2",
                "param",
                "set",
                node_name,
                param_name,
                value_str,
            ]
            rc_set, out_set = run_cmd(cmd_set, timeout_sec=timeout)
            if rc_set == 0:
                cmd_get = ["ros2", "param", "get", node_name, param_name]
                rc_get, out_get = run_cmd(cmd_get, timeout_sec=timeout)
                if rc_get == 0:
                    got = self._parse_ros_bool_param_get(out_get)
                    if got is None:
                        print(
                            f"  - WARN {node_name}.{param_name}: gesetzt, aber Rückleseformat unbekannt. "
                            "Akzeptiere als erfolgreich."
                        )
                        return True
                    if got == target_value:
                        return True
                    print(
                        f"  - WARN {node_name}.{param_name}: Rücklese-Wert={got}, "
                        f"erwartet={target_value} (Versuch {attempt}/{retries})"
                    )
                else:
                    print(
                        f"  - WARN {node_name}.{param_name}: set ok, get fehlgeschlagen "
                        f"(Versuch {attempt}/{retries})"
                    )
            else:
                short = out_set.splitlines()[-1] if out_set else "(kein output)"
                print(
                    f"  - WARN {node_name}.{param_name}: set fehlgeschlagen "
                    f"(Versuch {attempt}/{retries}) -> {short}"
                )

            if attempt < retries and retry_wait > 0.0:
                time.sleep(retry_wait)

        return False

    def _set_camera_param_bool(self, param_name: str, target_value: bool) -> bool:
        return self._set_node_param_bool(
            node_name=self.args.camera_node,
            param_name=param_name,
            target_value=target_value,
            retries=int(self.args.camera_param_retries),
            timeout=float(self.args.camera_param_timeout),
            retry_wait=float(self.args.camera_param_retry_wait),
        )

    def set_camera_processing_mode(self, mode_name: str, enabled: bool) -> bool:
        target = {
            "align_depth.enable": enabled,
        }
        print(
            f"\n[camera] Setze Modus '{mode_name}' auf Node {self.args.camera_node}: "
            f"align_depth.enable = {enabled}"
        )

        all_ok = True
        for name, val in target.items():
            ok = self._set_camera_param_bool(name, val)
            print(f"  - {name}: {'ok' if ok else 'FEHLER'}")
            all_ok = all_ok and ok

        if all_ok and self.args.camera_param_settle > 0.0:
            time.sleep(float(self.args.camera_param_settle))

        if not all_ok:
            print(
                "[camera] Konnte nicht alle Kamera-Parameter setzen. "
                "Aktion wird aus Sicherheitsgründen abgebrochen."
            )
        return all_ok

    def set_camera_align_depth_mode(self, enabled: bool) -> bool:
        print(
            f"\n[camera] Setze {self.args.camera_node}.align_depth.enable = {enabled}"
        )

        # Avoid unnecessary runtime reconfigure: RealSense can be unstable on no-op sets.
        cmd_get = ["ros2", "param", "get", self.args.camera_node, "align_depth.enable"]
        rc_get, out_get = run_cmd(cmd_get, timeout_sec=float(self.args.camera_param_timeout))
        if rc_get == 0:
            current = self._parse_ros_bool_param_get(out_get)
            if current is not None and current == enabled:
                print(f"  - align_depth.enable: bereits {enabled}, kein Set nötig")
                return True

        ok = self._set_camera_param_bool("align_depth.enable", enabled)
        print(f"  - align_depth.enable: {'ok' if ok else 'FEHLER'}")
        if not ok:
            print(
                "[camera] Konnte align_depth.enable nicht setzen. "
                "Aktion wird aus Sicherheitsgründen abgebrochen."
            )
        return ok

    def set_keypoint_depth_roi_mode(self, enabled: bool) -> bool:
        print(
            f"\n[keypoint] Setze {self.args.keypoint_node}.use_depth_roi = {enabled}"
        )
        ok = self._set_node_param_bool(
            node_name=self.args.keypoint_node,
            param_name="use_depth_roi",
            target_value=enabled,
            retries=int(self.args.keypoint_param_retries),
            timeout=float(self.args.keypoint_param_timeout),
            retry_wait=float(self.args.keypoint_param_retry_wait),
        )
        print(f"  - use_depth_roi: {'ok' if ok else 'FEHLER'}")

        if ok and self.args.keypoint_param_settle > 0.0:
            time.sleep(float(self.args.keypoint_param_settle))

        if not ok:
            print(
                "[keypoint] Konnte use_depth_roi nicht setzen. "
                "Aktion wird aus Sicherheitsgründen abgebrochen."
            )
        return ok

    def start_tracking(self) -> None:
        init_done = self.read_init_done()
        if init_done is not True:
            print("\n[tracking] Nicht erlaubt: Initialisierung ist noch nicht abgeschlossen.")
            return

        if not self._is_core_running():
            print("\n[tracking] Core läuft nicht, starte zuerst core nodes.")
            self.start_core()
            time.sleep(1.0)

        if not self.set_keypoint_depth_roi_mode(enabled=False):
            return

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

        # Always ensure aligned depth exists for initialization masking.
        if not self.set_camera_align_depth_mode(enabled=True):
            return

        if self.args.camera_runtime_mode_switch:
            # Initialization mode: use full depth processing chain
            if not self.set_camera_processing_mode("initialization", enabled=True):
                return
        else:
            print(
                "\n[camera] Initialisierungs-Mode-Switch deaktiviert "
                "(--camera-runtime-mode-switch nicht gesetzt); "
                "Post-Init-Set auf Tracking-Parameter bleibt aktiv."
            )
        if not self.set_keypoint_depth_roi_mode(enabled=True):
            return

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
            print("[init] Setze Kamera direkt auf Tracking-Parameter (Filter/Align aus) ...")
            if not self.set_camera_processing_mode("post_init_tracking_prep", enabled=False):
                return
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

    parser.add_argument("--input-topic", default="/camera/camera/color/image_raw/compressed")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw/compressedDepth")
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

    parser.add_argument("--camera-node", default="/camera/camera")
    parser.add_argument("--camera-param-timeout", type=float, default=6.0)
    parser.add_argument("--camera-param-retries", type=int, default=3)
    parser.add_argument("--camera-param-retry-wait", type=float, default=0.75)
    parser.add_argument("--camera-param-settle", type=float, default=1.0)
    parser.add_argument(
        "--camera-runtime-mode-switch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, toggle RealSense align_depth.enable at runtime "
            "for initialization mode."
        ),
    )

    parser.add_argument("--keypoint-node", default="/keypoint_node")
    parser.add_argument("--keypoint-param-timeout", type=float, default=6.0)
    parser.add_argument("--keypoint-param-retries", type=int, default=3)
    parser.add_argument("--keypoint-param-retry-wait", type=float, default=0.5)
    parser.add_argument("--keypoint-param-settle", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manager = IbvsSessionManager(args)
    return manager.run()


if __name__ == "__main__":
    sys.exit(main())
