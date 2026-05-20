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
- `download.chunking`: strategia di raggruppamento delle richieste CDS (`monthly` o `yearly`)
- `datasets.*`: collezioni CDS, variabili e orari da estrarre
- `paths.*`: cache, output e database SQLite locale
- `storage.*`: formato, compressione e naming file
- `logging.*`: livello, output su console, directory dei log e naming del file dedicato per ogni run
- `storage.intermediate_output`: `full` per mantenere gli intermedi correnti, `climatology` per salvare solo `fwi` e `mask` negli output mensili
- `processing.*`: resume, soglia terra/mare, buffer costiero opzionale, limiti operativi
- `percentile.*`: baseline multiannuale per il prodotto finale P90, mesi da includere, percentile richiesto, blocchi spaziali e naming dell'output finale

Uso di `download.chunking`:

- `monthly`: comportamento storico; ogni finestra mensile di processing genera una richiesta CDS dedicata
- `yearly`: raggruppa il download per anno di calendario, ma mantiene processing, checkpoint e output su base mensile
- con `yearly` il riuso della cache aumenta: tutte le finestre mensili dello stesso anno condividono lo stesso GRIB/NetCDF scaricato per dataset
- con `run` e `resume`, il calcolo resta limitato a `period.start` e `period.end`; per mantenere una singola richiesta annuale, il download puo' includere anche i mesi di bordo completi dell'anno interessato
- con `run-percentile`, `yearly` limita ogni richiesta annuale ai mesi definiti in `percentile.months`, sempre all'interno del singolo anno di baseline
- con `run-percentile`, `yearly` richiede che `percentile.months` definisca un intervallo contiguo, ad esempio `[5, 6, 7, 8, 9]`; configurazioni come `[5, 7, 9]` non sono supportate in questa modalita'
- con `run-percentile` e `yearly`, lo `spinup_days` della lavorazione annuale viene azzerato per non allargare il download oltre i mesi stagionali richiesti; se serve mantenere uno spinup precedente al primo mese selezionato, usare `monthly`

Esempio di configurazione per `yearly`:

```yaml
download:
	chunking: yearly
	remote_area_subset: false
	retry_attempts: 4
	retry_wait_seconds: 30
```

Sezione `logging` consigliata:

```yaml
logging:
	enabled: true
	level: INFO
	console: true
	file: true
	directory: data/state/logs
	filename_template: "{command}_{started_at}_{run_id}.log"
```

Con questa configurazione ogni invocazione `run`, `resume`, `run-percentile` o `aggregate-percentile` scrive lo stato di avanzamento sia a console sia in un file dedicato nella directory configurata. Il file contiene il tracciamento granulare delle richieste/download CDS, preprocess, clip, mask, loop giornaliero FWI, checkpoint, output e aggregazione percentile.

### CLI

Validazione configurazione:

```powershell
uv run fwi-module validate-config examples/greece.yaml
```

Esecuzione pipeline:

```powershell
uv run fwi-module run examples/greece.yaml
```

Override del livello di log e della directory dei file per-run:

```powershell
uv run fwi-module run examples/greece.yaml --log-level DEBUG --log-dir logs/fwi
```

Esecuzione su una sottofinestra temporale senza modificare il file YAML:

```powershell
uv run fwi-module run examples/greece.yaml --start 2023-04-01 --end 2023-04-07 --no-resume
```

Ripresa di un job interrotto:

```powershell
uv run fwi-module resume examples/greece.yaml
```

Workflow completo per costruire il raster finale multiyear P90 sui giorni maggio-settembre:

```powershell
uv run fwi-module run-percentile examples/greece.yaml
```

Aggregazione finale del raster P90 a partire dagli output mensili gia' presenti:

```powershell
uv run fwi-module aggregate-percentile examples/greece.yaml
```

Al termine di ogni comando di elaborazione il CLI stampa il percorso del file log generato, cosi' il run puo' essere monitorato in tempo reale a console e poi ispezionato a posteriori dal file dedicato.

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

Se `storage.intermediate_output: climatology`, gli output mensili intermedi vengono alleggeriti e contengono solo:

- `fwi`
- `mask`

Gli stati intermedi per il resume sono scritti come NetCDF4 separati nella directory `state_dir`.

I log runtime sono scritti in file separati sotto `logging.directory`; per default il percorso e' `state_dir/logs` e il nome del file include comando, timestamp UTC e `run_id`, in modo che ogni esecuzione abbia una traccia indipendente.

Quando e' configurata la sezione `percentile`, il package puo' produrre anche un raster finale con una variabile `fwi_pXX` che assegna a ogni cella il percentile richiesto dei valori giornalieri `fwi` calcolati sui mesi selezionati e su tutta la baseline multiannuale. Il file finale viene scritto nella stessa `output_dir` degli output mensili, con un nome derivato da `percentile.output_template`.

Per aumentare la copertura costiera si possono combinare:

- `processing.land_sea_threshold`: con `0.0` viene tenuto qualunque pixel con una frazione di terra positiva
- `processing.coastal_buffer_cells`: espande il mask finale di N celle, utile per includere meglio le coste anche prendendo alcuni pixel di mare

Gli output includono una variabile `spatial_ref` con metadati CF/GDAL in EPSG:4326 e una vera griglia geografica regolare: le coordinate `x`/`lon` sono in gradi est, le coordinate `y`/`lat` sono in gradi nord e i dati vengono riproiettati dalla Lambert conforme nativa di CERRA. Se i metadati CRS/proiezione nativa non sono presenti nel dataset sorgente, il package assume la proiezione Lambert standard di CERRA prima di eseguire la riproiezione verso 4326.

### Strategia prestazionale

- `monthly`: download e processing a finestre mensili
- `yearly`: download raggruppato per anno di calendario, processing e output ancora mensili
- per CERRA il crop remoto via `area` e' disattivato di default; il package scarica il raw file e ritaglia localmente sul bbox configurato
- riuso della cache locale per richieste identiche
- con `yearly` piu' finestre mensili dello stesso anno riusano la stessa entry di cache per dataset
- checkpoint per finestra completata
- calcolo FWI sequenziale nel tempo ma vettorizzato nello spazio
- per il prodotto multiyear P90 il processing viene eseguito stagione per stagione e anno per anno, con cataloghi di resume separati per annualita'; se `download.chunking: yearly`, ogni annualita' scarica una sola volta per dataset e limita la richiesta ai mesi configurati in `percentile.months`
- l'aggregazione finale legge solo blocchi spaziali 2D della variabile `fwi` dai NetCDF mensili, evitando di materializzare in RAM tutta la serie 20/30 anni
- per run climatologici lunghi su PC con 16GB di RAM e' consigliato impostare `storage.intermediate_output: climatology`; in questo modo gli intermedi mensili salvano solo `fwi` e `mask`, riducendo I/O e spazio disco senza cambiare la logica del percentile finale

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
