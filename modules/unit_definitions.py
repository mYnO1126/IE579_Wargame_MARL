# unit_definitions.py

from __future__ import annotations  # from .map import Coord

from enum import Enum
from collections import namedtuple
import numpy as np
import math




# Probability distributions for firing times
def triangular_distribution(M, C):
    return lambda x: np.random.triangular(M - C, M, M + C)


# def normal_distribution(mean, variance):
#     return np.random.normal(mean, np.sqrt(variance))


# def uniform_distribution(a, b):
#     return np.random.uniform(a, b)


# def constant_distribution(value):
#     return value



def constant_dist_func(value):
    return lambda x: value
    

def exp_decay(range_limit, p_hit, decay_const):
    return lambda r: (p_hit if r <= range_limit else 0) * math.exp(-r / decay_const)



def direct_fire_pk_func(coeff=1.0):
    return lambda p: (
        HitState.CKILL
        if p < 0.7 * coeff
        else (HitState.MKILL if p < 0.9 * coeff else HitState.FKILL)
    )


def simple_pk_func(pk):
    return lambda p: HitState.CKILL if p < pk else HitState.MISS


def indirect_pk_func():
    return lambda p, pk: HitState.CKILL if p < pk else HitState.MISS


def cookie_cutter_damage_func():
    """
    포탄의 착탄 지점과 목표물 간의 거리와 치명적 반경을 고려하여 피해를 계산하는 함수
    :return: 피해 확률
    """
    return lambda landing_distance, lethal_area_radius: (
        1.0 if landing_distance <= lethal_area_radius else 0.0
    )  # 피해 없음


def carlton_damage_func():
    """
    포탄의 착탄 지점과 목표물 간의 거리와 치명적 반경을 고려하여 피해를 계산하는 함수
    :return: 피해 확률
    """
    return lambda landing_distance, lethal_area_radius: (
        math.exp(-(landing_distance**2) / (lethal_area_radius**2))
        if landing_distance <= lethal_area_radius * 4.47
        else 0.0
    )  # 피해 없음


def gaussian_damage_func(
    b: float = 1.0,
):
    """
    포탄의 착탄 지점과 목표물 간의 거리와 치명적 반경을 고려하여 피해를 계산하는 함수
    :param b: 가우시안 분포의 표준편차
    :return: 피해 확률
    """
    return lambda landing_distance, lethal_area_radius: math.exp(
        -(landing_distance**2) / (2 * (b**2))
    )


def exponential_damage_func(
    b: float = 1.0,
):
    """
    포탄의 착탄 지점과 목표물 간의 거리와 치명적 반경을 고려하여 피해를 계산하는 함수
    :param b: 지수 분포의 매개변수
    :return: 피해 확률
    """
    return lambda landing_distance, lethal_area_radius: math.exp(-landing_distance / b)


def sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, size=1):
    """
    이변량 정규분포에서 무작위 (x, y) 샘플을 추출 (rho=0)
    :param mu_x: X축 평균
    :param mu_y: Y축 평균
    :param sigma_x: X축 표준편차
    :param sigma_y: Y축 표준편차
    :param size: 추출할 샘플 수
    :return: shape=(size, 2)의 numpy 배열
    """
    mean = [mu_x, mu_y]
    cov = [[sigma_x**2, 0], [0, sigma_y**2]]  # ρ = 0 ⇒ 공분산 = 0
    return np.random.multivariate_normal(mean, cov, size)


class UnitType(Enum):
    TANK = "tank"  # 전차
    MORTAR = "mortar"  # 박격포
    HOWITZER = "howitzer"  # 견인포
    SPG = "spg"  # 자주포
    MLRS = "mlrs"  # 다연장로켓포
    ATGM = "atgm"  # 대전차미사일
    RPG = "rpg"  # 휴대용 대전차 로켓포
    RECOILLESS = "recoilless"  # 무반동포
    # INFANTRY_AT = "infantry_at"  # 보병 대전차
    INFANTRY = "infantry"  # 보병
    SUPPLY = "supply"  # 보급차량
    # VEHICLE = "vehicle"  # 차량
    APC = "apc"  # 장갑차
    # DIR_FIRE_UNIT = ["tank", "atgm", "infantry_at"]
    # INDIRECT_FIRE_UNIT = ["mortar", "howitzer", "spg", "mlrs"]
    # ANTI_TANK = ["atgm", "recoilless", "rpg"]

    @classmethod
    def is_anti_tank(cls, unit_type):
        return unit_type in {
            cls.ATGM,
            cls.RPG,
            cls.RECOILLESS,
        }
    
    @classmethod
    def is_direct_fire(cls, unit_type):
        return unit_type in {
            cls.TANK,
            cls.ATGM,
            cls.RPG,
            cls.RECOILLESS,
        }
    
    @classmethod
    def is_indirect_fire(cls, unit_type):
        return unit_type in {
            cls.MORTAR,
            cls.HOWITZER,
            cls.SPG,
            cls.MLRS,
        }
    
    @classmethod
    def is_infantry(cls, unit_type):
        return unit_type in {
            cls.INFANTRY,
        }
    
    @classmethod
    def is_supply(cls, unit_type):
        return unit_type in {
            cls.SUPPLY,
        }


# class UnitCategory(Enum):
#     DIRECT_FIRE = "direct_fire"  # 직사화력

class UnitStatus(Enum):
    ALIVE = "alive"  # 살아있음
    DESTROYED = "destroyed"  # 파괴됨
    DAMAGED_MOBILITY = "mobility_damaged"  # 기동불능
    DAMAGED_FIREPOWER = "firepower_damaged"  # 화력불능
    OUT_OF_RANGE = "out_of_range"  # 사거리 초과
    MOVING = "moving"  # 이동중
    STATIONARY = "stationary"  # 정지중
    RELOADING = "reloading"  # 재장전중
    SPOTTED = "spotted"  # 발견됨
    UNSPOTTED = "unspotted"  # 발견되지 않음
    HIDDEN = "hidden"  # 은폐됨
    UNCOVERED = "uncovered"  # 은폐되지 않음
    ENGAGED = "engaged"  # 교전중
    UNENGAGED = "unengaged"  # 비교전중
    MOVEMENT_ORDER = "movement_order"  # 이동명령
    ENGAGEMENT_ORDER = "engagement_order"  # 교전명령
    RELOAD_ORDER = "reload_order"  # 재장전명령
    SPOT_ORDER = "spot_order"  # 발견명령
    UNSPOT_ORDER = "unspot_order"  # 발견되지 않음 명령
    HIDE_ORDER = "hide_order"  # 은폐명령
    UNCOVER_ORDER = "uncover_order"  # 은폐되지 않음 명령


class AmmoStatus(Enum):
    FULL = "full"  # 완전
    LOW = "low"  # 부족
    EMPTY = "empty"  # 없음


class UnitAction(Enum):
    MOVE = "move"  # 이동
    FIRE = "fire"  # 발사
    SPOT = "spot"  # 발견
    UNSPOT = "unspot"  # 발견되지 않음
    HIDE = "hide"  # 은폐
    UNCOVER = "uncover"  # 은폐되지 않음
    RELOAD = "reload"  # 재장전
    ENGAGE = "engage"  # 교전
    UNENGAGE = "unengage"  # 비교전
    SUPPLY = "supply"  # 보급
    REPAIR = "repair"  # 수리


class HitState(Enum):
    CKILL = "catastrophic-kill"  # 완전파괴
    MKILL = "mobility-kill"  # 기동불능
    FKILL = "firepower-kill"  # 화력불능
    MISS = "miss"  # 명중하지 않음


# 유닛 세부 정보를 담을 구조체
TroopCategory = namedtuple("TroopCategory", ["blue", "red"])


class UnitComposition(Enum):
    TANK = TroopCategory(blue={"Sho't_Kal": 10}, red={"T-55": 10, "T-62": 10})

    AT_WEAPON = TroopCategory(
        blue={"BGM-71_TOW": 12, "106mm_M40_Recoilless_Rifle": 36, "M72_LAW": 12},
        red={"9M14_Malyutka": 54, "107mm_B-11_Recoilless_Rifle": 36, "RPG-7": 54},
    )

    # TANK = TroopCategory(
    #     blue={"Sho't_Kal": 170},
    #     red={"T-55": 300, "T-62": 200}
    # )

    # # APC = TroopCategory( #TODO: 장갑차 추가, 사격 확률
    # #     blue={"M113": 20},
    # #     red={"BMP/BTR": 200}
    # # )
    # # INFANTRY = TroopCategory(    #TODO: 보병 추가
    # #     blue={"Golani×2 + ATGM중대": 850},
    # #     red={"보병여단3 + 기계화여단3": 4800}
    # # )
    # ARTILLERY = TroopCategory(
    #     blue={"60mm_Mortar": 12, "105mm_Howitzer": 20},
    #     red={"122mm_SPG": 200, "BM-21_MLRS": 200}  # "발" 단위는 맥락상 자주포 수량과 통합 처리
    # )

    # AT_WEAPON = TroopCategory(
    #     blue={"BGM-71_TOW": 12, "106mm_M40_Recoilless_Rifle": 36, "M72_LAW": 12},
    #     red={"9M14_Malyutka": 54, "107mm_B-11_Recoilless_Rifle": 36, "RPG-7": 54}
    # )
    # SUPPLY = TroopCategory(
    #     blue={"Blue_Supply_Truck": 40},
    #     red={"Red_Supply_Truck": 60}
    # )


class UnitSpec:
    def __init__(
        self,
        name,
        team,
        unit_type,
        range_km,
        ph_func,
        pk_func,
        damage_func=None,
        target_delay_func=constant_dist_func(2.0),
        fire_time_func=constant_dist_func(1.0),
        speed_road_kmh=30,
        speed_offroad_kmh=15,
    ):
        self.name = name
        self.team = team  # "blue" or "red"
        self.unit_type = unit_type  # UnitType Enum
        self.range_km = range_km
        self.ph_func = ph_func  # A function that returns hit probability
        self.pk_func = pk_func  # function that returns HitState
        self.damage_func = damage_func  # function that returns damage probability
        self.target_delay_func = target_delay_func
        self.fire_time_func = fire_time_func

        self.speed_road_kmh = speed_road_kmh
        self.speed_offroad_kmh = speed_offroad_kmh


UNIT_SPECS = {  # TODO: unit ph_func, pk_func 추가
    "Sho't_Kal": UnitSpec(
        name="Sho't_Kal",
        team="blue",
        unit_type=UnitType.TANK,
        range_km=2.5,
        ph_func=exp_decay(3.5, 0.8, 2.5),
        pk_func=direct_fire_pk_func(),
        target_delay_func=triangular_distribution(2.0, 1.0),
        fire_time_func=constant_dist_func(0.8),
        speed_road_kmh=35,
        speed_offroad_kmh=20
    ),
    "T-55": UnitSpec(
        name="T-55",
        team="red",
        unit_type=UnitType.TANK,
        range_km=2.0,
        ph_func=exp_decay(2.0, 0.7, 2.0),
        pk_func=direct_fire_pk_func(),
        target_delay_func=triangular_distribution(3.0, 1.0),
        fire_time_func=constant_dist_func(1.0),
        speed_road_kmh=50,
        speed_offroad_kmh=25
    ),
    "T-62": UnitSpec(
        name="T-62",
        team="red",
        unit_type=UnitType.TANK,
        range_km=2.0,
        ph_func=exp_decay(2.0, 0.68, 2.0),
        pk_func=direct_fire_pk_func(),
        target_delay_func=triangular_distribution(3.0, 1.0),
        fire_time_func=constant_dist_func(1.0),
        speed_road_kmh=50,
        speed_offroad_kmh=30
    ),
    "60mm_Mortar": UnitSpec(
        name="60mm_Mortar",
        team="blue",
        unit_type=UnitType.MORTAR,
        range_km=2.0,
        ph_func=None,
        pk_func=indirect_pk_func(),
        damage_func=gaussian_damage_func(b=3.0),
        target_delay_func=triangular_distribution(1.0, 0.5),
        fire_time_func=constant_dist_func(0.8),
        speed_road_kmh=0.00001,
        speed_offroad_kmh=3, #0.00001
    ),
    "105mm_Howitzer": UnitSpec(
        name="105mm_Howitzer",
        team="blue",
        unit_type=UnitType.HOWITZER,
        range_km=8.0, #전술적 우위가 너무 심해서 사거리 조정함. 11.0,
        ph_func=None,
        pk_func=indirect_pk_func(),
        damage_func=cookie_cutter_damage_func(),
        target_delay_func=triangular_distribution(15.0, 5.0), #전술적 우위가 너무 심해서 조정. triangular_distribution(3.0, 1.0),
        fire_time_func=constant_dist_func(15.0), #전술적 우위가 너무 심해서 조정. constant_dist_func(1.0),
        speed_road_kmh=0.00001,
        speed_offroad_kmh=0.00001
    ),
    "122mm_SPG": UnitSpec(
        name="122mm_SPG",
        team="red",
        unit_type=UnitType.SPG,
        range_km=15.0,
        ph_func=None,
        pk_func=indirect_pk_func(),
        damage_func=gaussian_damage_func(b=1.0),
        target_delay_func=triangular_distribution(3.0, 1.0),
        fire_time_func=constant_dist_func(1.0),
    ),
    "BM-21_MLRS": UnitSpec(
        name="BM-21_MLRS",
        team="red",
        unit_type=UnitType.MLRS,
        range_km=20.0,
        ph_func=None,
        pk_func=indirect_pk_func(),
        damage_func=exponential_damage_func(b=1.0),
        target_delay_func=triangular_distribution(1.0, 0.5),
        fire_time_func=constant_dist_func(0.05),
    ),
    "BGM-71_TOW": UnitSpec(
        name="BGM-71_TOW",
        team="blue",
        unit_type=UnitType.ATGM,
        range_km=3.75,
        ph_func=constant_dist_func(0.9),
        pk_func=direct_fire_pk_func(0.9),
        # pk_func=simple_pk_func(0.9),
        target_delay_func=triangular_distribution(1.0, 0.5),
        fire_time_func=constant_dist_func(1.5),
    ),
    "9M14_Malyutka": UnitSpec(
        name="9M14_Malyutka",
        team="red",
        unit_type=UnitType.ATGM,
        range_km=3.0,
        ph_func=constant_dist_func(0.85),
        pk_func=direct_fire_pk_func(0.85),
        # pk_func=simple_pk_func(0.85),
        target_delay_func=triangular_distribution(1.0, 0.5),
        fire_time_func=constant_dist_func(1.3),
        # speed_road_kmh=0.00001,
        # speed_offroad_kmh= 0.00001
    ),
    "106mm_M40_Recoilless_Rifle": UnitSpec(
        name="106mm_M40_Recoilless_Rifle",
        team="blue",
        unit_type=UnitType.RECOILLESS,
        range_km=1.2,
        ph_func=constant_dist_func(0.8),
        pk_func=direct_fire_pk_func(),
        target_delay_func=triangular_distribution(2.0, 0.5),
        fire_time_func=constant_dist_func(1.2),
    ),
    "107mm_B-11_Recoilless_Rifle": UnitSpec(
        name="107mm_B-11_Recoilless_Rifle",
        team="red",
        unit_type=UnitType.RECOILLESS,
        range_km=0.6,
        ph_func=constant_dist_func(0.75),
        pk_func=direct_fire_pk_func(),
        target_delay_func=triangular_distribution(2.0, 0.5),
        fire_time_func=constant_dist_func(1.0),
        # speed_road_kmh=0.00001,
        # speed_offroad_kmh=0.00001
    ),
    "M72_LAW": UnitSpec(
        name="M72_LAW",
        team="blue",
        unit_type=UnitType.RPG,
        range_km=0.3,
        ph_func=constant_dist_func(0.6),
        pk_func=direct_fire_pk_func(0.6),
        # pk_func=simple_pk_func(0.6),
        target_delay_func=triangular_distribution(1.0, 0.5),
        fire_time_func=constant_dist_func(0.8), #TODO: 재장전 없음
        # speed_road_kmh=0.00001,
        # speed_offroad_kmh=0.00001
    ),
    "RPG-7": UnitSpec(
        name="RPG-7",
        team="red",
        unit_type=UnitType.RPG,
        range_km=0.5,
        ph_func=constant_dist_func(0.65),
        pk_func=direct_fire_pk_func(0.65),
        # pk_func=simple_pk_func(0.65),
        target_delay_func=triangular_distribution(0.5, 0.17),
        fire_time_func=constant_dist_func(0.7),
        # speed_road_kmh=0.00001,
        # speed_offroad_kmh=0.00001       
    ),
    "M113": UnitSpec(
        name="M113",
        team="blue",
        unit_type=UnitType.APC,
        range_km=0.5,
        ph_func=constant_dist_func(0.30),
        pk_func=direct_fire_pk_func(), #TODO temp "exp(-r/2.0)",  # TODO
        target_delay_func=triangular_distribution(2.0, 1.0),
        fire_time_func=constant_dist_func(0.1),
        speed_road_kmh=64,
        speed_offroad_kmh=np.random.randint(40, 45),
    ),
    "BMP-1": UnitSpec(
        name="BMP-1",
        team="red",
        unit_type=UnitType.APC,
        range_km=0.8,
        ph_func=constant_dist_func(0.60),
        pk_func=direct_fire_pk_func(), #TODO temp # TODO
        target_delay_func=triangular_distribution(2.0, 1.0),
        fire_time_func=constant_dist_func(1.0),
        speed_road_kmh=65,
        speed_offroad_kmh=45,
    ),
    "AK-47": UnitSpec(
        name="AK-47",
        team="red",
        unit_type=UnitType.INFANTRY,
        range_km=0.3,
        ph_func=constant_dist_func(0.2),
        pk_func=direct_fire_pk_func(), #TODO constant_dist_func(0.25),  # TODO
        target_delay_func=triangular_distribution(2.0, 1.0),
        fire_time_func=constant_dist_func(0.1),
        speed_road_kmh=5,
        speed_offroad_kmh=5,
    ),
    # "BTR-60": UnitSpec(
    #     name="BTR-60",
    #     team="red",
    #     unit_type=UnitType.APC,
    #     range_km=0.8,
    #     ph_func=constant_dist_func(0.60),
    #     pk_func="exp(-r/2.0)", # TODO
    # target_delay_func=triangular_distribution(2.0, 1.0),
    #     fire_time_func=constant_dist_func(1.0),
    #     speed_road_kmh=80,
    #     speed_offroad_kmh=50
    # ),
    # "Golani×2 + ATGM중대": UnitSpec(
    #     name="Golani×2 + ATGM중대",
    #     team="blue",
    #     unit_type=UnitType.INFANTRY,
    #     range_km=0.3,  # 예: AK-47 유효사거리 0.3km
    #     ph_func=constant_dist_func(0.2),  # 예: Ph=0.2 at 300m
    #     pk_func="exp(-r/0.3)",
    #     target_delay_func=triangular_distribution(2.0, 1.0),
    #     fire_time_func=constant_dist_func(1.0),
    #     speed_road_kmh=5,
    #     speed_offroad_kmh=5,
    # ),
    # "보병여단3 + 기계화여단3": UnitSpec(
    #     name="보병여단3 + 기계화여단3",
    #     team="red",
    #     unit_type=UnitType.INFANTRY,
    #     range_km=0.3,
    #     ph_func=constant_dist_func(0.2),
    #     pk_func="exp(-r/0.3)",
    #     target_delay_func=triangular_distribution(2.0, 1.0),
    #     fire_time_func=constant_dist_func(1.0),
    #     speed_road_kmh=5,
    #     speed_offroad_kmh=5,
    # ),
    "Blue_Supply_Truck": UnitSpec(
        name="Blue_Supply_Truck",
        team="blue",
        unit_type=UnitType.SUPPLY,
        range_km=0.0,
        ph_func=exp_decay(0.1, 0.6, 0.1),
        pk_func="exp(-r/0.1)",
        target_delay_func=triangular_distribution(2.0, 1.0),
        fire_time_func=constant_dist_func(1.0),
        speed_road_kmh=80,
        speed_offroad_kmh=40,
    ),
    "Red_Supply_Truck": UnitSpec(
        name="Red Supply_Truck",
        team="red",
        unit_type=UnitType.SUPPLY,
        range_km=0.0,
        ph_func=exp_decay(0.1, 0.6, 0.1),
        pk_func="exp(-r/0.1)",
        target_delay_func=triangular_distribution(2.0, 1.0),
        fire_time_func=constant_dist_func(1.0),
        speed_road_kmh=80,
        speed_offroad_kmh=40,
    ),
}

curved_traj_weapon_data = {
    "60mm_Mortar": {
        "ballistics": [
            {
                "range_m": 1000,
                "flight_time": 10,
                "angle": 30,
                "aim_error_x": 10,
                "aim_error_r": 25,
                "ballistic_error_x": 7,
                "ballistic_error_r": 10,
            },
            {
                "range_m": 2000,
                "flight_time": 20,
                "angle": 60,
                "aim_error_x": 12,
                "aim_error_r": 29,
                "ballistic_error_x": 9,
                "ballistic_error_r": 16,
            },
        ],
        "lethal_area": [
            {"angle": 30, "open": 180, "forest": 90, "urban": 90},
            {"angle": 60, "open": 200, "forest": 100, "urban": 100},
            {"angle": 90, "open": 220, "forest": 110, "urban": 110},
        ],
    },
    "105mm_Howitzer": {
        "ballistics": [
            {
                "range_m": 3000,
                "flight_time": 18,
                "angle": 40,
                "aim_error_x": 15,
                "aim_error_r": 35,
                "ballistic_error_x": 12,
                "ballistic_error_r": 18,
            },
            {
                "range_m": 6000,
                "flight_time": 30,
                "angle": 36,
                "aim_error_x": 18,
                "aim_error_r": 50,
                "ballistic_error_x": 14,
                "ballistic_error_r": 26,
            },
            {
                "range_m": 9000,
                "flight_time": 40,
                "angle": 32,
                "aim_error_x": 20,
                "aim_error_r": 65,
                "ballistic_error_x": 16,
                "ballistic_error_r": 34,
            },
            {
                "range_m": 11000,
                "flight_time": 50,
                "angle": 30,
                "aim_error_x": 22,
                "aim_error_r": 80,
                "ballistic_error_x": 18,
                "ballistic_error_r": 40,
            },
        ],
        "lethal_area": [
            {"angle": 30, "open": 320, "forest": 160, "urban": 160},
            {"angle": 60, "open": 340, "forest": 170, "urban": 170},
            {"angle": 90, "open": 360, "forest": 180, "urban": 180},
        ],
    },
    "122mm_SPG": {
        "ballistics": [
            {
                "range_m": 5000,
                "flight_time": 22,
                "angle": 40,
                "aim_error_x": 18,
                "aim_error_r": 45,
                "ballistic_error_x": 14,
                "ballistic_error_r": 30,
            },
            {
                "range_m": 10000,
                "flight_time": 36,
                "angle": 35,
                "aim_error_x": 21,
                "aim_error_r": 60,
                "ballistic_error_x": 17,
                "ballistic_error_r": 38,
            },
            {
                "range_m": 15000,
                "flight_time": 50,
                "angle": 30,
                "aim_error_x": 24,
                "aim_error_r": 75,
                "ballistic_error_x": 20,
                "ballistic_error_r": 46,
            },
        ],
        "lethal_area": [
            {"angle": 30, "open": 380, "forest": 190, "urban": 190},
            {"angle": 60, "open": 400, "forest": 200, "urban": 200},
            {"angle": 90, "open": 420, "forest": 210, "urban": 210},
        ],
    },
    "BM-21_MLRS": {
        "ballistics": [
            {
                "range_m": 10000,
                "flight_time": 28,
                "angle": 50,
                "aim_error_x": 30,
                "aim_error_r": 90,
                "ballistic_error_x": 25,
                "ballistic_error_r": 45,
            },
            {
                "range_m": 15000,
                "flight_time": 38,
                "angle": 45,
                "aim_error_x": 35,
                "aim_error_r": 120,
                "ballistic_error_x": 28,
                "ballistic_error_r": 60,
            },
            {
                "range_m": 20000,
                "flight_time": 48,
                "angle": 40,
                "aim_error_x": 40,
                "aim_error_r": 150,
                "ballistic_error_x": 31,
                "ballistic_error_r": 75,
            },
        ],
        "lethal_area": [
            {"angle": 30, "open": 400, "forest": 200, "urban": 200},
            {"angle": 60, "open": 420, "forest": 210, "urban": 210},
            {"angle": 90, "open": 440, "forest": 220, "urban": 220},
        ],
    },
}


def interpolate_ballistics(weapon_name: str, target_range: float, weapon_data: dict):
    entries = weapon_data.get(weapon_name, {}).get("ballistics", [])
    if not entries:
        raise ValueError(f"No ballistics data found for weapon: {weapon_name}")

    # 정렬 보장 (range_m 기준)
    entries = sorted(entries, key=lambda e: e["range_m"])

    # 범위 밖 extrapolation 방지
    if target_range <= entries[0]["range_m"]: #TODO: 사거리 안쪽은 보간 유무 결정
        return entries[0]
        # raise ValueError(f"Target out of range for weapon: {weapon_name}")
    if target_range >= entries[-1]["range_m"]:
        raise ValueError(f"Target out of range for weapon: {weapon_name}")

    # 보간할 구간 찾기
    for i in range(len(entries) - 1):
        low = entries[i]
        high = entries[i + 1]
        if low["range_m"] <= target_range <= high["range_m"]:
            ratio = (target_range - low["range_m"]) / (high["range_m"] - low["range_m"])
            interpolated = {
                "range_m": target_range,
                "flight_time": low["flight_time"]
                + ratio * (high["flight_time"] - low["flight_time"]),
                "angle": low["angle"] + ratio * (high["angle"] - low["angle"]),
                "aim_error_x": low["aim_error_x"]
                + ratio * (high["aim_error_x"] - low["aim_error_x"]),
                "aim_error_r": low["aim_error_r"]
                + ratio * (high["aim_error_r"] - low["aim_error_r"]),
                "ballistic_error_x": low["ballistic_error_x"]
                + ratio * (high["ballistic_error_x"] - low["ballistic_error_x"]),
                "ballistic_error_r": low["ballistic_error_r"]
                + ratio * (high["ballistic_error_r"] - low["ballistic_error_r"]),
            }
            return interpolated

    raise ValueError(
        "Interpolation failed - this should never happen if data is valid."
    )


def interpolate_lethal_area(weapon_name: str, angle: float, weapon_data: dict):
    entries = weapon_data.get(weapon_name, {}).get("lethal_area", [])
    if not entries or len(entries) < 2:
        raise ValueError(f"Not enough lethal area data for weapon: {weapon_name}")

    # 정렬 (낙탄각 기준)
    entries = sorted(entries, key=lambda e: e["angle"])

    for i in range(len(entries) - 1):
        low = entries[i]
        high = entries[i + 1]

        if low["angle"] <= angle <= high["angle"]:
            ratio = (angle - low["angle"]) / (high["angle"] - low["angle"])
        elif angle < entries[0]["angle"]:
            low = entries[0]
            high = entries[1]
            ratio = (angle - low["angle"]) / (high["angle"] - low["angle"])
        elif angle > entries[-1]["angle"]:
            raise ValueError("Angle bigger than 90 degrees.")
        else:
            raise ValueError("Interpolation failed.")

        return {
            "angle": angle,
            "open": low["open"] + ratio * (high["open"] - low["open"]),
            "forest": low["forest"] + ratio * (high["forest"] - low["forest"]),
            "urban": low["urban"] + ratio * (high["urban"] - low["urban"]),
        }

    # fallback, shouldn't hit this
    raise ValueError("Interpolation failed.")


def get_shell_landing_point(
        target_coord, 
        aim_error_x, 
        aim_error_r, 
        ballistic_error_x, 
        ballistic_error_r
        ): # TODO: 오차 계산에 좌표계 변환 고려 필요
    """
    포탄의 착탄 지점을 계산하는 함수
    :param target_coord: 목표물의 좌표
    :param aim_error_x: 조준 오차 (x축)
    :param aim_error_r: 조준 오차 (y축)
    :param ballistic_error_x: 포탄 궤적 오차 (x축)
    :param ballistic_error_r: 포탄 궤적 오차 (y축)
    :return: 포탄의 착탄 지점 (x, y)
    """
    aim_x, aim_y = sample_bivariate_normal(0, 0, aim_error_x, aim_error_r, size=1)[0]
    ballistic_x, ballistic_y = sample_bivariate_normal(0, 0, ballistic_error_x, ballistic_error_r, size=1)[0]
    return target_coord.x + aim_x + ballistic_x, target_coord.y + aim_y + ballistic_y


def get_landing_data(
    weapon_name: str,
    target_coord,
    target_distance: float,
    target_environment: str = "open",
    ):
    """
    포탄의 궤적을 고려한 명중 확률을 계산하는 함수
    :param weapon_name: 무기 이름
    :param target_coord: 목표물의 좌표
    :param target_distance: 목표물과의 거리
    :param target_environment: 목표물의 환경 (open, forest, urban)
    :return: 포탄의 착탄 지점 (x, y)과 치명적 반경
    """
    weapon_data = curved_traj_weapon_data.get(weapon_name)
    if not weapon_data:
        raise ValueError(f"No data found for weapon: {weapon_name}")

    # 포탄의 궤적 데이터
    lethal_area = weapon_data["lethal_area"]

    interpolated_ballistic_data = interpolate_ballistics(
        weapon_name, target_distance, curved_traj_weapon_data
    )
    # 포탄의 착탄 지점 계산
    landing_x, landing_y = get_shell_landing_point(
        target_coord,
        interpolated_ballistic_data["aim_error_x"],
        interpolated_ballistic_data["aim_error_r"],
        interpolated_ballistic_data["ballistic_error_x"],
        interpolated_ballistic_data["ballistic_error_r"],
    )

    lethal_area_data = interpolate_lethal_area(
        weapon_name, interpolated_ballistic_data["angle"], curved_traj_weapon_data
    )
    if lethal_area_data[target_environment] is None:
        raise ValueError(f"No lethal area data found for env: {target_environment}")
    lethal_area = lethal_area_data[target_environment]
    lethal_area_radius = math.sqrt(lethal_area / math.pi)  # 원형으로 가정

    return landing_x, landing_y, lethal_area_radius


AmmunitionInfo = namedtuple("AmmunitionInfo", [
    "main_ammo",         # 주포 탄약 수량
    "secondary_ammo",    # 부무장 탄약 수량 (소총/기관총)
    "daily_main_usage",  # 주포 예상 일일 소모량
    "daily_sec_usage"    # 부무장 예상 일일 소모량
])

# 단위: 발 / 소모량은 1일 기준
AMMUNITION_DATABASE = {
    # 전차 및 장갑차
    "Sho't_Kal": AmmunitionInfo(main_ammo=64, secondary_ammo=4000, daily_main_usage=30, daily_sec_usage=2000),
    "T-55": AmmunitionInfo(main_ammo=43, secondary_ammo=3500, daily_main_usage=25, daily_sec_usage=1800),
    "T-62": AmmunitionInfo(main_ammo=40, secondary_ammo=2500, daily_main_usage=20, daily_sec_usage=1500),
    "M113": AmmunitionInfo(main_ammo=0, secondary_ammo=2000, daily_main_usage=0, daily_sec_usage=1000),
    "BMP-1": AmmunitionInfo(main_ammo=40, secondary_ammo=2000, daily_main_usage=20, daily_sec_usage=1000),
    "BTR-60": AmmunitionInfo(main_ammo=0, secondary_ammo=2000, daily_main_usage=0, daily_sec_usage=1000),

    # 대전차 무기
    "BGM-71_TOW": AmmunitionInfo(main_ammo=10, secondary_ammo=0, daily_main_usage=5, daily_sec_usage=0),
    "9M14_Malyutka": AmmunitionInfo(main_ammo=5, secondary_ammo=0, daily_main_usage=3, daily_sec_usage=0),
    "106mm_M40_Recoilless_Rifle": AmmunitionInfo(main_ammo=6, secondary_ammo=0, daily_main_usage=4, daily_sec_usage=0),
    "107mm_B-11_Recoilless_Rifle": AmmunitionInfo(main_ammo=5, secondary_ammo=0, daily_main_usage=3, daily_sec_usage=0),
    "M72_LAW": AmmunitionInfo(main_ammo=1, secondary_ammo=0, daily_main_usage=1, daily_sec_usage=0),
    "RPG-7": AmmunitionInfo(main_ammo=5, secondary_ammo=0, daily_main_usage=3, daily_sec_usage=0),

    # 보병 및 소총
    "AK-47": AmmunitionInfo(main_ammo=300, secondary_ammo=0, daily_main_usage=200, daily_sec_usage=0),

    # 박격포 / 자주포 / 로켓
    "60mm_Mortar": AmmunitionInfo(main_ammo=30, secondary_ammo=0, daily_main_usage=20, daily_sec_usage=0),
    "105mm_Howitzer": AmmunitionInfo(main_ammo=40, secondary_ammo=0, daily_main_usage=25, daily_sec_usage=0),
    "122mm_SPG": AmmunitionInfo(main_ammo=40, secondary_ammo=0, daily_main_usage=30, daily_sec_usage=0),
    "BM-21_MLRS": AmmunitionInfo(main_ammo=40, secondary_ammo=0, daily_main_usage=40, daily_sec_usage=0),
}

#공급 수량
SUPPLY_DATABASE = {
    "Sho't_Kal": 128,
    "T-55": 129,
    "T-62": 129,
    "RPG-7": 120,
    "106mm_M40_Recoilless_Rifle": 120,
    "BGM-71_TOW": 100,
    "BM-21_MLRS": 120,
    "105mm_Howitzer": 120,
    "122mm_SPG": 120,
    "60mm_Mortar": 120,
    "9M14_Malyutka": 100,
    "107mm_B-11_Recoilless_Rifle": 100,
    "M72_LAW": 36,
    "AK-47": 12000,
    "M113": 12000,
}



if __name__ == "__main__":
    # # Test the interpolation function
    # weapon_name = "60mm_Mortar"
    # target_range = 1500
    # interpolated_data = interpolate_ballistics(
    #     weapon_name, target_range, curved_traj_weapon_data
    # )
    # print(f"Interpolated data for {weapon_name} at {target_range}m: {interpolated_data}")
    print(interpolate_lethal_area("BM-21_MLRS", 35, curved_traj_weapon_data))  # 보간
    print(interpolate_lethal_area("BM-21_MLRS", 20, curved_traj_weapon_data))  # 하한 외삽
    # print(interpolate_lethal_area("BM-21_MLRS", 100, curved_traj_weapon_data))