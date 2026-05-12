## fwi-module

Package Python per il calcolo del Canadian Fire Weather Index (FWI) su griglia geografica giornaliera per la Grecia, usando i dataset CERRA del Climate Data Store (CDS) di ECMWF/Copernicus.

### Stato attuale

La base del package e' gia' implementata:

- configurazione YAML validata con Pydantic
- CLI minima per validazione config, esecuzione e ispezione cache
- motore FWI giornaliero vettorizzato su griglia
- download layer con `ecmwf-datastores-client` e fallback `cdsapi`
- cache file-based piu' catalogo SQLite per download e processing windows
- checkpoint NetCDF4 per resume dopo interruzioni
- preprocessing dei file scaricati e output NetCDF4 per training ML/DL

### Requisiti

- Python 3.12+
- progetto gestito con `uv`
- credenziali CDS valide
- termini del dataset accettati sul portale CDS

### Installazione

Installazione base:

```powershell
uv sync
```

Installazione con dipendenze di sviluppo:

```powershell
uv sync --extra dev
```

Installazione con supporto GRIB:

```powershell
uv sync --extra grib
```

Installazione completa consigliata per esecuzione locale e test:

```powershell
uv sync --extra dev --extra grib
```

### Credenziali CDS

Il package non hardcoda credenziali. Le credenziali possono essere lette da:

- variabili d'ambiente `ECMWF_DATASTORES_URL` e `ECMWF_DATASTORES_KEY`
- file di configurazione protetto `~/.ecmwfdatastoresrc`
- fallback `cdsapi` con `.cdsapirc`

Esempio di file `~/.ecmwfdatastoresrc`:

```yaml
url: https://cds.climate.copernicus.eu/api
key: your-personal-access-token
```

Prima di usare il package:

1. creare un account sul CDS
2. generare il token personale
3. accettare i Terms of Use dei dataset CERRA usati

### Dataset usati

La configurazione di default usa:

- `reanalysis-cerra-single-levels` per temperatura, umidita' relativa e vento a 10 m
- `reanalysis-cerra-land` per precipitazione giornaliera e land-sea mask

Non e' implementato un fallback automatico verso dataset alternativi se CERRA non copre le date richieste: il package fallisce con un errore esplicito.

### Configurazione YAML

Un esempio completo e' disponibile in `examples/greece.yaml`.

Campi principali:

- `region.bbox`: bounding box `[north, west, south, east]`
- `period.start`, `period.end`: periodo richiesto
- `period.spinup_days`: giorni extra prima dell'inizio per stabilizzare FFMC, DMC e DC
- `datasets.*`: collezioni CDS, variabili e orari da estrarre
- `paths.*`: cache, output e database SQLite locale
- `storage.*`: formato, compressione e naming file
- `processing.*`: resume, maschera terra/mare, limiti operativi

### CLI

Validazione configurazione:

```powershell
uv run fwi-module validate-config examples/greece.yaml
```

Esecuzione pipeline:

```powershell
uv run fwi-module run examples/greece.yaml
```

Esecuzione su una sottofinestra temporale senza modificare il file YAML:

```powershell
uv run fwi-module run examples/greece.yaml --start 2023-04-01 --end 2023-04-07 --no-resume
```

Ripresa di un job interrotto:

```powershell
uv run fwi-module resume examples/greece.yaml
```

Ispezione cache/catalogo:

```powershell
uv run fwi-module inspect-cache examples/greece.yaml
```

### Output

Gli output finali sono scritti in NetCDF4 compresso, con variabili:

- `ffmc`
- `dmc`
- `dc`
- `isi`
- `bui`
- `fwi`
- `dsr`

Se `storage.include_inputs` e' attivo, il file contiene anche gli input meteorologici preprocessati:

- `temperature`
- `relative_humidity`
- `wind_speed`
- `precipitation`
- `mask`

Gli stati intermedi per il resume sono scritti come NetCDF4 separati nella directory `state_dir`.

Gli output includono una variabile `spatial_ref` con metadati CF/GDAL e coordinate geografiche ausiliarie `lat`/`lon`, in modo che strumenti GIS come QGIS possano riconoscere il riferimento spaziale del file.

### Strategia prestazionale

- download e processing a finestre mensili
- per CERRA il crop remoto via `area` e' disattivato di default; il package scarica il raw file e ritaglia localmente sul bbox configurato
- riuso della cache locale per richieste identiche
- checkpoint per finestra completata
- calcolo FWI sequenziale nel tempo ma vettorizzato nello spazio

### Note operative

- CDS/CERRA fornisce normalmente i raw data in GRIB; il package li converte e salva gli output finali in NetCDF4.
- La maschera della Grecia usa confini Natural Earth tramite `regionmask` e si combina con la land-sea mask del dataset.
- Nei test end-to-end viene usato un backend fittizio: per esecuzioni reali servono credenziali e accesso al CDS.

### Test

Eseguire l'intera suite:

```powershell
uv run --extra dev pytest
```

### Prossimi step naturali

- affinare il mapping esatto delle request CDS contro il dataset scelto in produzione
- aggiungere smoke test con file CERRA reali
- valutare un export opzionale Zarr per pipeline distribuite
