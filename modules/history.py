# history.py

import pandas as pd
#!CLAUDE segfault 방지: 대화형 TkAgg 백엔드는 장시간 루프에서 figure를 반복 생성/저장할 때
#         간헐적 크래시(segfault)를 유발한다. 시뮬레이션은 PNG 저장 전용이므로 비대화형 Agg로 고정.
#         (pyplot import 이전에 backend를 지정해야 적용됨)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

from .unit_definitions import UnitType
from .troop import Troop, TroopList
import numpy as np
# from .map import MAP_WIDTH, MAP_HEIGHT


#!CLAUDE 전술 프레임 제목용: 시뮬레이션 분 → 시계 시각 문자열 (시작 13:55).
def _sim_clock_str(current_time):
    total = int(13 * 60 + 55 + current_time)
    day = total // 1440 + 1
    hour = (total % 1440) // 60
    minute = total % 60
    return f"Day {day}  {hour:02d}:{minute:02d}"


class History:  # Store history of troop actions and troop status
    def __init__(self, time):
        self.current_time = time
        self.battle_log = []
        self.status_data = {
            "time": [],
        }

        self.visualization_data = {"time": [], "unit": [], "x": [], "y": [], "z": []}

    def update_time(self, time):  # update current time
        if time >= self.current_time:
            self.current_time = time
        else:
            raise ValueError("Time cannot be set to a past value.")
        self.current_time = time

    def init_status_data(self, troop_list: TroopList, reference_altitude, height):  # initialize status data
        troops = troop_list.troops
        for troop in troops:
            if f"{troop.id}_status" not in self.status_data:
                self.status_data[f"{troop.id}_status"] = []
                self.status_data[f"{troop.id}_target"] = []
                self.status_data[f"{troop.id}_fire_time"] = []
        self.add_to_status_data(troop_list, reference_altitude, height)

    def add_to_battle_log(
        self, type_, shooter, target, target_type, result
    ):  # add to battle log
        self.battle_log.append(
            [self.current_time, shooter, type_, target, target_type, result]
        )

    def add_to_status_data(self, troop_list: TroopList, reference_altitude, height):  # add to status data
        troops = troop_list.troops
        troop_ids = troop_list.troop_ids

        self.status_data["time"].append(self.current_time)
        troop_dict = {t.id: t for t in troops}
        s_data = self.status_data

        for tid in troop_ids:
            status_key = s_data[f"{tid}_status"]
            target_key = s_data[f"{tid}_target"]
            fire_key = s_data[f"{tid}_fire_time"]

            troop = troop_dict.get(tid)
            if troop:
                status_key.append(troop.status.value)
                target_key.append(troop.target.id if troop.target else None)
                fire_key.append(troop.next_fire_time if troop.alive else None)
            else:
                status_key.append("destroyed")
                target_key.append(None)
                fire_key.append(None)

        for troop in troops:
            if troop.alive:
                self.visualization_data["time"].append(self.current_time)
                self.visualization_data["unit"].append(troop.id)
                self.visualization_data["x"].append(troop.coord.x) # x -> 가로축
                # self.visualization_data["y"].append(troop.coord.y)
                # self.visualization_data["z"].append(troop.coord.z) 
                self.visualization_data["y"].append(troop.coord.z - reference_altitude) # y -> 높이
                self.visualization_data["z"].append(height - troop.coord.y) # z -> 세로축

    def get_battle_log(self):  # return battle log
        return self.battle_log

    def get_status_data(self):  # return status data
        return self.status_data

    def save_battle_log(self, foldername="res/res0"):  # save battle log to file
        columns = ["time", "shooter", "shooter_type", "target", "target_type", "result"]
        df = pd.DataFrame(self.battle_log, columns=columns)
        df.to_csv(foldername + "/battle_log.csv", index=False)
        print("Battle log saved to battle_log.csv")

    def save_status_data(self, foldername="res/res0"):  # save status data to file
        df = pd.DataFrame(self.status_data)
        df.to_csv(foldername + "/status_data.csv", index=False)
        print("Status data saved to status_data.csv")

        df_2 = pd.DataFrame(self.visualization_data)
        df_2.to_csv(foldername + "/visualization_data.csv", index=False)
        print("Visualization data saved to visualization_data.csv")

    def save_status_data_new(
        self, troop_list, battle_map, filename="status_data.csv"
    ):  # save status data to file
        data = []
        for t_idx, time in enumerate(self.status_data["time"]):
            time_sec = round(time * 60, 2)  # Convert from minutes to seconds
            for troop in troop_list:
                if troop.id in self.status_data:
                    # Get current position
                    x = troop.coord.x # x -> 가로축
                    # y = troop.coord.y
                    # z = troop.coord.z
                    y = troop.coord.z - battle_map.reference_altitude # y -> 높이
                    z = battle_map.height - troop.coord.y # z -> 세로축
                    data.append([time_sec, troop.id, x, y, z])

        df = pd.DataFrame(data, columns=["time", "unit", "x", "y", "z"])
        df.to_csv(filename, index=False)
        print("Status data saved to status_data.csv")

    def draw_troop_positions(self, Map, troop_list, current_time, save_dir="frames",
                           show_attack_lines=True, show_ranges=True, show_paths=False,
                           show_move_arrows=False, show_legend=False):
        # plt.figure(figsize=(16, 8))

        # --- 입력 데이터 ---
        def binarize(mask):
            return (mask > 0).astype(int)

        dem_arr = Map.dem_arr
        road_mask = binarize(Map.road_mask)
        lake_mask = binarize(Map.lake_mask)
        wood_mask = binarize(Map.wood_mask)
        stream_mask = binarize(Map.stream_mask)

        H, W = dem_arr.shape
        road_mask = road_mask[:H, :W]
        lake_mask = lake_mask[:H, :W]
        wood_mask = wood_mask[:H, :W]
        stream_mask = stream_mask[:H, :W]

        # --- 컬러맵 정의 ---
        road_cmap = ListedColormap(['none', 'red'])
        lake_cmap = ListedColormap(['none', 'blue'])
        wood_cmap = ListedColormap(['none', 'green'])
        stream_cmap = ListedColormap(['none', 'purple'])

        # --- 시각화 ---
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(dem_arr, cmap="terrain", origin="upper", alpha = 0.2)

        ax.imshow(road_mask, cmap=road_cmap, alpha=0.6, origin="upper")
        ax.imshow(lake_mask, cmap=lake_cmap, alpha=0.5, origin="upper")
        ax.imshow(wood_mask, cmap=wood_cmap, alpha=0.5, origin="upper")
        ax.imshow(stream_mask, cmap=stream_cmap, alpha=0.5, origin="upper")
        # --- 지형 시각화 추가 ---

        # 🟢 1. 사거리 원 그리기 (공격선보다 먼저 그려서 뒤에 위치)
        if show_ranges:
            self._draw_weapon_ranges(ax, troop_list)

        # 🟢 2. 공격선 그리기
        if show_attack_lines:
            self._draw_attack_lines(ax, troop_list)

        # 🟢 3. 이동 경로 그리기 (선택사항)
        if show_paths:
            self._draw_movement_paths(ax, troop_list)

        # 🟢 4. 부대 위치 그리기 (맨 위에 표시)
        self._draw_troop_markers(ax, troop_list)

        #!CLAUDE 실제 이동 방향(velocity) 화살표. 기본 OFF(기존 프레임 불변), RL 렌더에서만 ON.
        if show_move_arrows:
            self._draw_move_arrows(ax, troop_list)

        #!CLAUDE 버그 수정: TroopList는 iterable이 아님 → .troops 를 순회해야 함 (--save_frames 사용 시 TypeError로 즉시 크래시 방지).
        # for troop in troop_list:
        for troop in troop_list.troops:
            if not troop.alive:
                continue
            color = "blue" if troop.team == "blue" else "red"
            color = "grey" if not troop.active else color
            marker = "o" if troop.type == UnitType.TANK else "s"
            plt.scatter(
                np.float32(troop.coord.x),
                np.float32(troop.coord.y),
                c=color,
                marker=marker,
                label=troop.id,
                alpha=0.7,
                s=10,
            )
        plt.title(f"Troop Positions at T={current_time:.0f} min")
        # plt.xlim(0, MAP_WIDTH)
        # plt.ylim(0, MAP_HEIGHT)
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.grid(True)
        # --- 설정 ---

        # ----지형 시각화 추가 ----
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_title(f"Tactical Situation Board - T={current_time:.0f}Min", fontsize=14, fontweight='bold')
        ax.set_xlabel("X (10m)")
        ax.set_ylabel("Y (10m)")
        ax.grid(True, alpha=0.3)

        # 범례 (지형용)
        legend_elements = [
            Patch(facecolor='red', edgecolor='r', label='Road'),
            Patch(facecolor='blue', edgecolor='b', label='Lake'),
            Patch(facecolor='green', edgecolor='g', label='River'),
            Patch(facecolor='purple', edgecolor='purple', label='Stream'),
            Patch(facecolor='blue', edgecolor='k', label='Blue Troop'),
            Patch(facecolor='red', edgecolor='k', label='Red Troop'),
            # 부대
            mpatches.Circle((0,0), 1, facecolor='blue', edgecolor='k', label='Blue'),
            mpatches.Circle((0,0), 1, facecolor='red', edgecolor='k', label='Red'),
            mpatches.Circle((0,0), 1, facecolor='grey', edgecolor='k', label='Inactive'),
            # 전술 요소
            plt.Line2D([0], [0], color='red', linewidth=2, alpha=0.7, label='FireLine'),
            mpatches.Circle((0,0), 1, facecolor='none', edgecolor='orange', 
                          alpha=0.3, label='AttackRange'),
        ]

        if show_paths:
            legend_elements.append(
                plt.Line2D([0], [0], color='cyan', linewidth=1,
                          linestyle='--', alpha=0.6, label='Path')
            )
        #!CLAUDE 이동 방향 화살표 범례
        if show_move_arrows:
            legend_elements.append(
                plt.Line2D([0], [0], color='black', marker='>', linestyle='-',
                          linewidth=1, label='Move dir')
            )

        #!CLAUDE 범례는 옵션(기본 OFF). show_legend=True 일 때만 표시.
        if show_legend:
            ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.2, 0.0))
        # ----지형 시각화 추가 ----

        plt.tight_layout()
        plt.savefig(f"{save_dir}/frame_{int(current_time):05d}.png")
        plt.close()

    def _draw_attack_lines(self, ax, troop_list: TroopList):
        """공격선 그리기"""
        for troop in troop_list.troops:
            if not troop.alive or not troop.active:
                continue
            
            if troop.target and troop.target.alive:
                # 공격선 색상 결정
                line_color = 'darkred' if troop.team == 'red' else 'darkblue'
                
                # 무기 유형별 선 스타일
                if UnitType.is_indirect_fire(troop.type):
                    # 간접화력: 곡선 스타일
                    linestyle = ':'
                    linewidth = 1.5
                    alpha = 0.6
                elif UnitType.is_anti_tank(troop.type):
                    # 대전차: 굵은 실선
                    linestyle = '-'
                    linewidth = 2.5
                    alpha = 0.8
                elif troop.type == UnitType.TANK:
                    # 전차: 실선
                    linestyle = '-'
                    linewidth = 2.0
                    alpha = 0.7
                else:
                    # 기타: 얇은 실선
                    linestyle = '-'
                    linewidth = 1.0
                    alpha = 0.5

                # 공격선 그리기
                ax.plot([troop.coord.x, troop.target.coord.x],
                       [troop.coord.y, troop.target.coord.y],
                       color=line_color, linestyle=linestyle, 
                       linewidth=linewidth, alpha=alpha)
                
                # 🟢 화살표 추가 (공격 방향 표시)
                self._add_attack_arrow(ax, troop, line_color, alpha)

    def _add_attack_arrow(self, ax, troop, color, alpha):
        """공격 방향 화살표 추가"""
        dx = troop.target.coord.x - troop.coord.x
        dy = troop.target.coord.y - troop.coord.y
        
        # 화살표 크기 조정
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            # 타겟 근처에 화살표 배치 (80% 지점)
            arrow_x = troop.coord.x + 0.8 * dx
            arrow_y = troop.coord.y + 0.8 * dy
            
            # 화살표 크기 정규화
            arrow_dx = (dx / length) * 8  # 8픽셀 크기
            arrow_dy = (dy / length) * 8
            
            ax.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy,
                    head_width=3, head_length=4, 
                    fc=color, ec=color, alpha=alpha, linewidth = 1)

    def _draw_weapon_ranges(self, ax, troop_list: TroopList):
        """무기 사거리 원 그리기"""
        for troop in troop_list.troops:
            if not troop.alive or not troop.active:
                continue
            
            if troop.range_km > 0:
                # 사거리를 픽셀로 변환 (1km = 100픽셀)
                range_pixels = troop.range_km * 100
                
                # 무기 유형별 색상
                if UnitType.is_indirect_fire(troop.type):
                    color = 'purple'
                    alpha = 0.15
                elif UnitType.is_anti_tank(troop.type):
                    color = 'orange'
                    alpha = 0.2
                elif troop.type == UnitType.TANK:
                    color = 'yellow'
                    alpha = 0.2
                else:
                    color = 'gray'
                    alpha = 0.1
                
                # 사거리 원 그리기
                circle = plt.Circle((troop.coord.x, troop.coord.y), 
                                  range_pixels, 
                                  fill=False, edgecolor=color, 
                                  alpha=alpha, linewidth=1)
                ax.add_patch(circle)

    def _draw_movement_paths(self, ax, troop_list: TroopList):
        """이동 경로 그리기"""
        for troop in troop_list.troops:
            if not troop.alive or not troop.can_move:
                continue
            
            # 경로가 있는 경우
            if hasattr(troop, 'path') and troop.path:
                path_x = [troop.coord.x] + [p[0] for p in troop.path]
                path_y = [troop.coord.y] + [p[1] for p in troop.path]
                
                ax.plot(path_x, path_y, 
                       color='cyan', linestyle='--', 
                       linewidth=1, alpha=0.6)
            
            # 고정 목적지가 있는 경우
            elif troop.fixed_dest:
                ax.plot([troop.coord.x, troop.fixed_dest.x],
                       [troop.coord.y, troop.fixed_dest.y],
                       color='lime', linestyle='-.', 
                       linewidth=1, alpha=0.5)

    def _draw_troop_markers(self, ax, troop_list: TroopList, size_scale=1.0):
        """부대 마커 그리기"""
        #!CLAUDE size_scale 인자 추가: 전술 개요(넓은 맵)에서 마커를 키워 가독성↑. 기본 1.0이면 기존 동작 동일.
        for troop in troop_list.troops:
            if not troop.alive:
                continue
            
            # 색상 결정
            if not troop.active:
                color = "grey"
                alpha = 0.5
            else:
                color = "blue" if troop.team == "blue" else "red"
                alpha = 0.8
            
            # 마커 모양 결정
            if troop.type == UnitType.TANK:
                marker = "o"
                size = 25
            elif troop.type == UnitType.APC:
                marker = "s"
                size = 20
            elif UnitType.is_indirect_fire(troop.type):
                marker = "^"
                size = 20
            elif UnitType.is_anti_tank(troop.type):
                marker = "D"
                size = 15
            else:
                marker = "s"
                size = 10
            
            ax.scatter(troop.coord.x, troop.coord.y,
                      c=color, marker=marker,
                      s=size * size_scale, alpha=alpha,
                      edgecolors='black', linewidths=0.5)

    #!CLAUDE 실제 이동 방향 화살표(velocity 기반). 고정 길이로 그려 속도와 무관하게 방향만 표시.
    def _draw_move_arrows(self, ax, troop_list: TroopList):
        """각 유닛이 실제로 움직이는 방향(velocity)을 검은 화살표로 표시."""
        for troop in troop_list.troops:
            if not troop.alive or not getattr(troop, "active", False):
                continue
            vx, vy = troop.velocity.x, troop.velocity.y
            speed = float(np.hypot(vx, vy))
            if speed < 1e-6:
                continue   # 정지 중인 유닛은 화살표 없음
            ux, uy = vx / speed, vy / speed
            L = 10.0  # 고정 화살 길이(px)
            ax.arrow(troop.coord.x, troop.coord.y, ux * L, uy * L,
                     head_width=4.0, head_length=4.5, fc="black", ec="black",
                     alpha=0.95, linewidth=1.6, zorder=6, length_includes_head=True)

    def create_tactical_overview(self, Map, troop_list: TroopList, current_time, save_dir="frames",
                                 show_legend=False):
        """🟢 전술 개요 시각화 (별도 파일)"""
        #!CLAUDE 가독성 개선: 지형 배경 + 팀/병종 범례 + 시계 제목, 교전 히트맵 정비.
        #         + 프레임 안정화: 축 위치를 고정(add_axes)하고 컬러바 자리를 항상 예약해
        #           프레임마다 레이아웃/크기가 흔들리거나 축 라벨이 잘리던 문제 제거.
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        #
        # # 좌측: 전체 전술 상황
        # self._create_overview_plot(ax1, Map, troop_list, current_time)
        #
        # # 우측: 교전 강도 히트맵
        # self._create_engagement_heatmap(ax2, Map, troop_list, current_time)
        #
        # plt.tight_layout()
        # plt.savefig(f"{save_dir}/tactical_{int(current_time):05d}.png", dpi=150)
        # plt.close()

        fig = plt.figure(figsize=(18, 8))
        fig.patch.set_facecolor("white")

        # 축 위치를 고정(figure 비율 좌표) → 모든 프레임이 동일한 레이아웃으로 저장됨.
        # cax(컬러바 자리)는 교전이 없어도 항상 예약해 두어 좌/우 패널이 절대 이동하지 않게 한다.
        ax1 = fig.add_axes([0.045, 0.10, 0.40, 0.80])   # 좌: 전술 상황
        ax2 = fig.add_axes([0.545, 0.10, 0.36, 0.80])   # 우: 교전 히트맵
        cax = fig.add_axes([0.915, 0.10, 0.013, 0.80])  # 히트맵 컬러바 자리

        self._create_overview_plot(ax1, Map, troop_list, current_time, show_legend=show_legend)
        self._create_engagement_heatmap(ax2, Map, troop_list, current_time, cax=cax)

        fig.suptitle(
            f"Tactical Overview   —   T = {int(current_time)} min   ({_sim_clock_str(current_time)})",
            fontsize=16, fontweight="bold", y=0.97,
        )
        fig.savefig(f"{save_dir}/tactical_{int(current_time):05d}.png",
                    dpi=150, facecolor="white")
        plt.close(fig)

    def _create_overview_plot(self, ax, Map, troop_list: TroopList, current_time, show_legend=False):
        """전체 전술 상황 플롯"""
        #!CLAUDE 가독성 개선: 단조로운 DEM 대신 지형 마스크(도로/물/숲/개울)를 함께 표시하고
        #         부대 마커를 키우며 팀·병종·지형 범례를 추가.
        # # 간단한 지형 표시
        # ax.imshow(Map.dem_arr, cmap="terrain", origin="upper", alpha=0.3)
        #
        # # 부대 위치만 표시 (공격선 없이)
        # self._draw_troop_markers(ax, troop_list)
        #
        # ax.set_title(f"Overall Situation - T={current_time:.0f} min")
        # ax.set_xlim(0, Map.width)
        # ax.set_ylim(Map.height, 0)

        def binarize(mask):
            return (mask > 0).astype(int)

        # 지형 배경 (고도 + 도로/숲/물/개울)
        ax.imshow(Map.dem_arr, cmap="terrain", origin="upper", alpha=0.35)
        ax.imshow(binarize(Map.road_mask),   cmap=ListedColormap(["none", "#8a5a2b"]), origin="upper", alpha=0.55)
        ax.imshow(binarize(Map.wood_mask),   cmap=ListedColormap(["none", "#2e8b57"]), origin="upper", alpha=0.40)
        ax.imshow(binarize(Map.lake_mask),   cmap=ListedColormap(["none", "#2f6fb0"]), origin="upper", alpha=0.55)
        ax.imshow(binarize(Map.stream_mask), cmap=ListedColormap(["none", "#5aa0e0"]), origin="upper", alpha=0.55)

        #!CLAUDE 전술 요소 추가: 공격선(사격 방향) + goal선(이동 목표)을 마커 뒤에 깔고,
        #         이동 방향 화살표는 마커 위에. (이전엔 마커만 그려 방향 정보가 안 보였음)
        self._draw_attack_lines(ax, troop_list)
        self._draw_movement_paths(ax, troop_list)

        # 부대 마커 (크게)
        self._draw_troop_markers(ax, troop_list, size_scale=4.0)

        # 실제 이동 방향 화살표 (마커 위)
        self._draw_move_arrows(ax, troop_list)

        ax.set_title("Tactical Situation", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("X (px · 10 m)", fontsize=10)
        ax.set_ylabel("Y (px · 10 m)", fontsize=10)
        ax.set_xlim(0, Map.width)
        ax.set_ylim(Map.height, 0)
        ax.set_aspect("equal")

        #!CLAUDE 범례는 옵션(기본 OFF) — show_legend=True 일 때만 두 범례를 그림
        if not show_legend:
            return

        # 범례 1: 팀 색상 + 지형 (좌상단)
        map_handles = [
            Patch(facecolor="blue", edgecolor="black", label="BLUE (Israel)"),
            Patch(facecolor="red", edgecolor="black", label="RED (Syria)"),
            Patch(facecolor="grey", edgecolor="black", label="Inactive"),
            Patch(facecolor="#8a5a2b", label="Road"),
            Patch(facecolor="#2e8b57", label="Woods"),
            Patch(facecolor="#2f6fb0", label="Water"),
        ]
        leg1 = ax.legend(handles=map_handles, loc="upper left", fontsize=8,
                         framealpha=0.85, title="Map", title_fontsize=9)
        ax.add_artist(leg1)

        # 범례 2: 병종 모양 (우상단)
        type_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#3a3a3a", markersize=9,  label="Tank"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#3a3a3a", markersize=8,  label="APC"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#3a3a3a", markersize=9,  label="Artillery"),
            plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#3a3a3a", markersize=7,  label="Anti-tank"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#3a3a3a", markersize=5,  label="Infantry / etc"),
        ]
        ax.legend(handles=type_handles, loc="upper right", fontsize=8,
                  framealpha=0.85, title="Unit type", title_fontsize=9)

    def _create_engagement_heatmap(self, ax, Map, troop_list: TroopList, current_time, cax=None):
        """교전 강도 히트맵"""
        # 맵 실제 크기 (하드코딩 800x600 대신 사용)
        W, H = Map.width, Map.height

        #!CLAUDE 가독성 개선: 옅은 지형 배경 위에 교전 밀도를 올리고(빈 셀은 투명),
        #         좌측 상황도와 동일한 좌표 방향(y 아래로 증가)·스타일·시계 제목으로 통일.
        # 옅은 지형 배경으로 공간 맥락 제공
        ax.imshow(Map.dem_arr, cmap="Greys", origin="upper", alpha=0.18)

        # 활성 교전 중인 부대들의 위치를 기반으로 히트맵 생성
        engagement_data = [
            [troop.coord.x, troop.coord.y]
            for troop in troop_list.troops
            if troop.alive and troop.active and troop.target
        ]

        ax.set_title("Engagement Intensity", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("X (px · 10 m)", fontsize=10)
        ax.set_ylabel("Y (px · 10 m)", fontsize=10)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)   # 좌측 상황도와 동일하게 y 아래로 증가
        ax.set_aspect("equal")

        if engagement_data:
            engagement_data = np.array(engagement_data)

            # 2D 히스토그램으로 교전 밀도 계산
            #!CLAUDE 하드코딩 800x600 → 맵 실제 크기(W,H)로 교체
            # bins=50, range=[[0, 800], [0, 600]]
            hist, xedges, yedges = np.histogram2d(
                engagement_data[:, 0], engagement_data[:, 1],
                bins=35, range=[[0, W], [0, H]]
            )

            #!CLAUDE 하드코딩 800x600 → 맵 실제 크기(W,H)로 교체 + 좌표 방향/컬러맵 정리
            # im = ax.imshow(hist.T, origin='lower', cmap='Reds', alpha=0.7,
            #               extent=[0, 800, 0, 600])
            # 낮은 값(빈 셀)은 set_under로 투명 처리. np.ma 마스킹 대신 이 방식을 쓰는 이유:
            # 가우시안 보간 시 마스크 경계가 흐려져 히트가 거의 안 보이게 되는 문제를 피하기 위함.
            heat_cmap = plt.get_cmap("inferno").copy()
            heat_cmap.set_under(color=(0, 0, 0, 0))  # 투명
            im = ax.imshow(hist.T, origin="lower", cmap=heat_cmap, vmin=0.8, alpha=0.9,
                           extent=[0, W, 0, H], interpolation="gaussian")

            # 컬러바는 고정 자리(cax)에 그려 프레임 간 레이아웃이 흔들리지 않게 함
            if cax is not None:
                cax.set_visible(True)
                cbar = plt.colorbar(im, cax=cax)
            else:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label("engaging units / cell", fontsize=9)
        else:
            # 교전이 없으면 컬러바 자리는 비워 둔다(다른 축은 고정이라 이동하지 않음)
            if cax is not None:
                cax.set_visible(False)
            ax.text(0.5, 0.5, "No Active Engagement",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=13, color="#888888", fontstyle="italic")


    def plot_team_strength_over_time(self, foldername="res/res0", show_plot=True):
        df = pd.DataFrame(self.status_data)

        time_col = df["time"]
        blue_cols = [
            col for col in df.columns if "_status" in col and col.startswith("B")
        ]
        red_cols = [
            col for col in df.columns if "_status" in col and col.startswith("R")
        ]

        blue_alive = df[blue_cols].apply(
            lambda row: sum(status == "alive" for status in row), axis=1
        )
        red_alive = df[red_cols].apply(
            lambda row: sum(status == "alive" for status in row), axis=1
        )

        #!CLAUDE 가독성: 저장 그래프 스타일 개선(팀 색상/영역 채움/최종 병력 주석/깔끔한 격자·여백).
        #         데이터 계산 로직은 위와 동일하며 그래프 모양만 변경.
        # plt.figure(figsize=(10, 5))
        # plt.plot(time_col, blue_alive, label="BLUE Troops Alive")
        # plt.plot(time_col, red_alive, label="RED Troops Alive")
        # plt.xlabel("Time (min)")
        # plt.ylabel("Number of Troops Alive")
        # plt.title("Team Strength Over Time")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(foldername + "/plot.png", dpi=300)  # ✅ 파일 저장
        # if show_plot:
        #     plt.show()
        # print(f"Graph saved as {foldername}/plot.png")

        BLUE = "#2b6cb0"   # 이스라엘(방어)
        RED = "#c0392b"    # 시리아(공격)

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fbfbfb")

        # 영역 채움 + 라인
        ax.fill_between(time_col, blue_alive, color=BLUE, alpha=0.12, zorder=1)
        ax.fill_between(time_col, red_alive, color=RED, alpha=0.12, zorder=1)
        ax.plot(time_col, blue_alive, color=BLUE, linewidth=2.4,
                label="BLUE (Israel)", zorder=3)
        ax.plot(time_col, red_alive, color=RED, linewidth=2.4,
                label="RED (Syria)", zorder=3)

        # 최종 병력 수를 선 끝에 표시
        if len(time_col) > 0:
            for series, color in ((blue_alive, BLUE), (red_alive, RED)):
                x_end = time_col.iloc[-1]
                y_end = series.iloc[-1]
                ax.scatter([x_end], [y_end], color=color, s=38,
                           zorder=5, edgecolors="white", linewidths=1.0)
                ax.annotate(f"{int(y_end)}", (x_end, y_end),
                            textcoords="offset points", xytext=(9, 0),
                            color=color, fontsize=11, fontweight="bold",
                            va="center")

        #!CLAUDE 야간(19:00~06:00) 구간 음영. 이동 로직(calculate_movement_distance)의 daynight 정의와 동일한
        #         시계 기준 사용: hour = (13*60+55 + t)//60 % 24, 야간 = hour < 6 or hour >= 19.
        #         (참고: 전투 로직의 is_night = 360 <= t%1440 <= 1080 과는 정의가 다름)
        START_MIN = 13 * 60 + 55  # 시뮬레이션 시작 시각 13:55 (분)

        def _is_night(t):
            hour = int((START_MIN + t) // 60) % 24
            return hour < 6 or hour >= 19

        if len(time_col) > 0:
            t0, t1 = int(time_col.min()), int(time_col.max())
            span_start = None
            night_labeled = False
            for t in range(t0, t1 + 1):
                if _is_night(t):
                    if span_start is None:
                        span_start = t
                elif span_start is not None:
                    ax.axvspan(span_start, t, color="#34495e", alpha=0.10, zorder=2,
                               label=None if night_labeled else "Night (19:00–06:00)")
                    night_labeled = True
                    span_start = None
            if span_start is not None:  # 마지막까지 야간으로 끝나는 경우
                ax.axvspan(span_start, t1, color="#34495e", alpha=0.10, zorder=2,
                           label=None if night_labeled else "Night (19:00–06:00)")

        # 축/제목/격자 스타일
        ax.set_title("Team Strength Over Time", fontsize=16, fontweight="bold", pad=12)
        ax.set_xlabel("Time (minutes from 13:55)", fontsize=11.5)
        ax.set_ylabel("Troops Alive", fontsize=11.5)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.02)
        ax.grid(True, color="#d4d4d4", linestyle="--", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.legend(frameon=False, fontsize=11.5, loc="lower left")

        fig.tight_layout()
        fig.savefig(foldername + "/plot.png", dpi=300, bbox_inches="tight")  # ✅ 파일 저장
        if show_plot:
            plt.show()
        plt.close(fig)  # figure 누수 방지
        print(f"Graph saved as {foldername}/plot.png")
