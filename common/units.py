"""Small unit helpers used across experiments."""
import math

def db_to_power_ratio(db: float) -> float:
    return 10 ** (db / 10)

def power_ratio_to_db(ratio: float) -> float:
    return 10 * math.log10(ratio)

def db_per_cm_to_alpha_per_um(db_per_cm: float) -> float:
    """Power attenuation coefficient alpha (1/um) from dB/cm."""
    return db_per_cm / 4.343 / 1e4

def alpha_per_um_to_db_per_cm(alpha_per_um: float) -> float:
    return alpha_per_um * 4.343 * 1e4

def delta_f_ghz_from_delta_lambda_nm(dlam_nm: float, lambda_nm: float) -> float:
    """Small-change rule: df = -(c/lambda^2) dlambda, returned as a positive GHz per positive nm."""
    c = 299_792_458.0
    return c * dlam_nm * 1e-9 / (lambda_nm * 1e-9) ** 2 / 1e9

def delta_lambda_nm_from_delta_f_ghz(df_ghz: float, lambda_nm: float) -> float:
    c = 299_792_458.0
    return df_ghz * 1e9 * (lambda_nm * 1e-9) ** 2 / c / 1e-9
