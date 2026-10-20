# CUDA - Sobel Optimized

Confronto tra una reference Sobel in Python puro e sei varianti PyCUDA ottimizzate (baseline, shared memory, constant memory, unrolling e block shape diverse) per l'edge detection su immagini a diverse risoluzioni.

## Cosa fa

Il progetto applica il filtro Sobel a immagini convertite in scala di grigi e salva la magnitudine del gradiente come PNG. La reference CPU è in Python puro; le varianti GPU usano PyCUDA con il kernel CUDA incorporato nello script e compilato a runtime.

L'obiettivo è didattico/sperimentale: confrontare correttezza e tempi delle diverse strategie di ottimizzazione, non fornire una libreria pronta per la produzione.

## Struttura dei file

| File | Ruolo |
| --- | --- |
| `cpu_sobel.py` | Reference CPU in Python puro |
| `gpu-2D2D.py` | Baseline GPU, filtri in global memory |
| `gpu-2D2D-SM.py` | Variante con shared memory |
| `gpu-2D2D-CM.py` | Variante con constant memory |
| `gpu-2D2D-CM-SM.py` | Constant memory + shared memory |
| `gpu-2D2D-CM-SM-UNR.py` | Shared memory + kernel 3x3 srotolato |
| `gpu-2D2D-CM-SM-UNR-DimDiverse.py` | Constant + shared + unrolled, con sweep di block shape |
| `*.spec` | Configurazioni PyInstaller per gli script GPU |
| `Dataset/` | Immagini di input locali, non versionate |
| `Output/` | Immagini generate, non versionate |
| `smoke_test.sh` | Test rapido su una sola immagine |
| `requirements.txt` | Dipendenze Python |

## Prerequisiti

- Python 3.10+ (l'ambiente attuale usa Python 3.12).
- GPU NVIDIA con driver installato; verifica con `nvidia-smi`.
- CUDA Toolkit con `nvcc` disponibile nel `PATH`; verifica con `nvcc --version`.
- Compilatore C++ per l'installazione di PyCUDA, che su PyPI è distribuito come sorgente.
- Dipendenze Python: `numpy`, `Pillow`, `pycuda`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -c "import pycuda.autoinit; print('PyCUDA OK')"
```

## Quick start

```bash
mkdir -p Dataset
cp /percorso/immagine.jpg Dataset/

python3 gpu-2D2D.py
ls Output/
```

`Dataset/` deve esistere e contenere almeno un'immagine `.png`, `.jpg` o `.jpeg`. `Output/` viene creata automaticamente. Gli script non accettano argomenti da riga di comando: percorsi e block size sono definiti nei rispettivi blocchi `__main__`.

Il branch `main` non versiona `Dataset/`; immagini campione sono disponibili nel branch `origin/Andrea`. Non committare immagini locali o output generati.

## Varianti

| Script | Ottimizzazione | Block size | Output esempio (`HD.jpg`) |
| --- | --- | --- | --- |
| `cpu_sobel.py` | Reference Python puro | n/a | `HD_sobel_cpu.png` |
| `gpu-2D2D.py` | Baseline, filtri in global memory | 16x16 | `HD_sobel_baseline_16x16.png` |
| `gpu-2D2D-SM.py` | Shared memory | 8x8, 16x16, 32x32, 1024x1 | `HD_sobel_sm_16x16.png` |
| `gpu-2D2D-CM.py` | Constant memory | 16x16 | `HD_sobel_cm_16x16.png` |
| `gpu-2D2D-CM-SM.py` | Constant memory + shared memory | 16x16 | `HD_sobel_cm-sm_16x16.png` |
| `gpu-2D2D-CM-SM-UNR.py` | Shared memory + 3x3 srotolato; `CM` solo nel nome | 16x16 | `HD_sobel_sm-unr_16x16.png` |
| `gpu-2D2D-CM-SM-UNR-DimDiverse.py` | Constant + shared + unrolled | 10 forme, da 32x16 a 1024x1 | `HD_sobel_cm-sm-unr_32x16.png` |

`gpu-2D2D-CM-SM-UNR.py` contiene `CM` nel nome per motivi storici, ma non usa la constant memory: i coefficienti sono scritti direttamente nel kernel.

## Output e naming

Formato per le varianti GPU:

```text
<stem>_sobel_<variant>_<W>x<H>.png
```

Per la CPU, senza block size:

```text
<stem>_sobel_cpu.png
```

Token delle varianti:

| Variante | Token |
| --- | --- |
| `cpu_sobel.py` | `cpu` |
| `gpu-2D2D.py` | `baseline` |
| `gpu-2D2D-SM.py` | `sm` |
| `gpu-2D2D-CM.py` | `cm` |
| `gpu-2D2D-CM-SM.py` | `cm-sm` |
| `gpu-2D2D-CM-SM-UNR.py` | `sm-unr` |
| `gpu-2D2D-CM-SM-UNR-DimDiverse.py` | `cm-sm-unr` |

La stessa variante con lo stesso blocco sovrascrive il file precedente: è voluto. Per conservare più serie di esperimenti, usare directory `Output/` separate.

## Rigenerare gli output

Per ripartire da zero, eliminare solo i PNG dentro `Output/` dopo averne verificato l'elenco, poi rilanciare gli script:

```bash
find Output -maxdepth 1 -type f -name '*.png' -print
# Se l'elenco è corretto:
find Output -maxdepth 1 -type f -name '*.png' -delete
```

La convenzione di naming evita collisioni tra baseline/CM e tra SM/CM-SM. La reference CPU è molto lenta su immagini grandi: usare immagini ridotte per i confronti CPU vs GPU.

## Risultati

Non ci sono benchmark Sobel versionati in questa repo. I CSV recuperati dal branch `origin/Andrea` (`report.csv` e `Report/RISULTATI -CPU(Foglio1).csv`) riguardano il vecchio esperimento di **rotazione** (colonna `Angle`) e non sono confrontabili con le varianti Sobel.

Riepilogo dei tempi CPU storici, con 8 angoli per immagine:

| Immagine | Risoluzione | Tempo medio (s) | Min (s) | Max (s) |
| --- | --- | ---: | ---: | ---: |
| `640x960.jpg` | 640x960 | 2.27 | 1.78 | 2.83 |
| `HD.jpg` | 1200x673 | 1.43 | 1.02 | 1.90 |
| `FullHD.jpg` | 1920x1080 | 7.09 | 2.84 | 9.72 |
| `4K.jpg` | 4368x2893 | 28.93 | 14.92 | 60.43 |
| `8K.jpg` | 4480x6720 | 107.63 | 82.85 | 136.37 |
| `Sopra 8k.jpg` | 6336x9520 | 191.89 | 71.89 | 272.38 |

Questi dati servono solo come archivio storico. Per ottenere tempi Sobel, eseguire gli script e annotare i valori stampati a terminale, ricordando che i timer misurano solo il kernel e non l'intera pipeline.

## Kernel duplicato

Il sorgente CUDA è duplicato in tutti i sei script GPU. Non esiste un modulo condiviso: se si modificano coefficienti Sobel, `clamp`, `sqrtf` o la gestione dell'halo, la modifica va replicata in ogni variante.

Ogni script GPU contiene un commento `KERNEL SYNC` come promemoria. L'estrazione opzionale delle stringhe CUDA in un futuro `kernels.py` è possibile, ma richiede di aggiornare import, spec PyInstaller e tutte le varianti: non fa parte del setup minimo.

## Build PyInstaller

Ogni script GPU ha uno spec dedicato:

```bash
pyinstaller --clean gpu-2D2D.spec
```

Se si rinomina uno script, aggiornare sia il percorso in `Analysis([...])` sia `name` nello spec. `gpu.spec` è un file legacy e punta a `gpu.py`, che non è più presente: non usarlo finché non viene aggiornato. L'eseguibile prodotto non è completamente autonomo: PyCUDA compila i kernel a runtime, quindi il sistema di destinazione deve avere driver NVIDIA e CUDA Toolkit/`nvcc`.

## Limiti noti

- La reference CPU usa quattro cicli Python annidati ed è molto lenta su immagini grandi.
- `cpu_sobel.py` va in errore con immagini di altezza inferiore a 10 pixel, a causa della divisione `height // 10` usata per il progresso.
- Le varianti con shared memory non inizializzano tutte le celle dell'halo fuori dall'immagine: il bordo di 1 pixel può differire dalla CPU o risultare non deterministico. Per i confronti usare i pixel interni.
- I timer misurano solo il lancio del kernel e `cuda.Context.synchronize()`: non includono allocazione, trasferimenti host-device/device-host e I/O su disco.
- Le prestazioni dipendono dalla GPU e dalla versione del toolkit CUDA; i block size migliori non sono universali.
- Non ci sono test automatici, CI o lint. La verifica disponibile è `smoke_test.sh` più il confronto visivo/strumentale degli output.
