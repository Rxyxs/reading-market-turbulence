"""Descarga reproducible de trades reales desde Binance Data Vision.

Baja los dumps diarios de aggTrades (ZIP), verifica el SHA256 que publica
Binance junto a cada archivo, y deja el CSV en data/raw/<SIMBOLO>/.
No requiere API key.

Es idempotente: un dia ya descargado y verificado se omite, asi que se puede
volver a correr para completar un rango sin bajar de nuevo lo que ya esta.

    python scripts/download.py --symbols BTCUSDT ETHUSDT --start 2026-08-21 --end 2026-08-25
    python scripts/download.py --symbols BTCUSDT --start 2026-08-25 --end 2026-08-25 --force
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
ROOT = Path(__file__).resolve().parent.parent
DEST_POR_DEFECTO = ROOT / "data" / "raw"
TIEMPO_LIMITE = 120


def dias(inicio: date, fin: date) -> list[date]:
    if fin < inicio:
        raise ValueError(f"la fecha final ({fin}) es anterior a la inicial ({inicio})")
    return [inicio + timedelta(days=i) for i in range((fin - inicio).days + 1)]


def nombre_archivo(simbolo: str, dia: date) -> str:
    return f"{simbolo}-aggTrades-{dia.isoformat()}"


def url_zip(simbolo: str, dia: date) -> str:
    return f"{BASE_URL}/{simbolo}/{nombre_archivo(simbolo, dia)}.zip"


def destino_csv(simbolo: str, dia: date, dest: Path) -> Path:
    return dest / simbolo / f"{nombre_archivo(simbolo, dia)}.csv"


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _descargar(url: str, destino: Path) -> None:
    """Descarga a un archivo temporal y renombra al final: un corte de red no deja
    un archivo a medias que la proxima corrida tomaria por bueno."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=TIEMPO_LIMITE) as respuesta:
        with tempfile.NamedTemporaryFile(dir=destino.parent, delete=False, suffix=".part") as temporal:
            parcial = Path(temporal.name)
            shutil.copyfileobj(respuesta, temporal)
    parcial.replace(destino)


def _sha_publicado(url: str) -> str | None:
    """Binance publica <archivo>.zip.CHECKSUM con 'sha256  nombre'. Si no existe, se avisa."""
    try:
        with urllib.request.urlopen(f"{url}.CHECKSUM", timeout=TIEMPO_LIMITE) as respuesta:
            return respuesta.read().decode().split()[0].lower()
    except urllib.error.URLError:
        return None


def descargar_dia(simbolo: str, dia: date, dest: Path, forzar: bool = False) -> tuple[Path, str]:
    """Devuelve (ruta del csv, estado) donde estado es 'omitido', 'descargado' o el error."""
    csv = destino_csv(simbolo, dia, dest)
    if csv.exists() and csv.stat().st_size > 0 and not forzar:
        return csv, "omitido"

    url = url_zip(simbolo, dia)
    zip_temporal = csv.with_suffix(".zip")
    try:
        _descargar(url, zip_temporal)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return csv, "no publicado (404): fecha futura, sin datos o simbolo inexistente"
        return csv, f"error HTTP {error.code}"
    except urllib.error.URLError as error:
        return csv, f"error de red: {error.reason}"

    esperado = _sha_publicado(url)
    if esperado is not None and sha256(zip_temporal) != esperado:
        zip_temporal.unlink(missing_ok=True)
        return csv, "SHA256 no coincide con el publicado por Binance"

    try:
        with zipfile.ZipFile(zip_temporal) as comprimido:
            internos = [n for n in comprimido.namelist() if n.lower().endswith(".csv")]
            if len(internos) != 1:
                zip_temporal.unlink(missing_ok=True)
                return csv, f"el zip trae {len(internos)} csv, se esperaba 1"
            with comprimido.open(internos[0]) as origen, open(csv, "wb") as salida:
                shutil.copyfileobj(origen, salida)
    except zipfile.BadZipFile:
        zip_temporal.unlink(missing_ok=True)
        return csv, "el archivo descargado no es un zip valido"
    finally:
        zip_temporal.unlink(missing_ok=True)

    aviso = "" if esperado is not None else " (sin CHECKSUM publicado, no verificado)"
    return csv, f"descargado{aviso}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="simbolos de Binance spot")
    parser.add_argument("--start", required=True, type=date.fromisoformat, help="primer dia, YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, help="ultimo dia inclusive (por defecto, el mismo que --start)")
    parser.add_argument("--dest", type=Path, default=DEST_POR_DEFECTO, help="raiz de datos crudos (data/raw)")
    parser.add_argument("--force", action="store_true", help="re-descarga aunque el csv ya exista")
    args = parser.parse_args(argv)

    fechas = dias(args.start, args.end or args.start)
    print(f"{len(args.symbols)} simbolo(s) x {len(fechas)} dia(s) -> {args.dest}")

    fallas = 0
    for simbolo in args.symbols:
        for dia in fechas:
            csv, estado = descargar_dia(simbolo, dia, args.dest, forzar=args.force)
            if estado.startswith(("descargado", "omitido")):
                tamano = csv.stat().st_size / 1e6 if csv.exists() else 0.0
                print(f"  {simbolo} {dia}: {estado} ({tamano:,.1f} MB)")
            else:
                fallas += 1
                print(f"  {simbolo} {dia}: {estado}", file=sys.stderr)

    if fallas:
        print(f"\n{fallas} descarga(s) fallaron", file=sys.stderr)
        return 1
    print(f"\nListo. Datos en {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
