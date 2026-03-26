#!/bin/bash
#SBATCH --job-name=PrepDatos_EU
#SBATCH --output=prep_datos_%j.out
#SBATCH --error=prep_datos_%j.err
#SBATCH --account=ehpc485
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00

# 1. Carga de modulos (ORDEN CORREGIDO: impi antes de hdf5)
module purge
module load intel
module load mkl
module load impi
module load hdf5
module load python/3.12.1

# 2. Arreglo del PYTHONPATH y entorno
unset PYTHONPATH
export PYTHONPATH="/gpfs/projects/ehpc485/tesi681824/transcriptor/venv_transcriptor/lib/python3.12/site-packages"
source /gpfs/projects/ehpc485/tesi681824/transcriptor/venv_transcriptor/bin/activate

# 3. Bloqueo de internet
export HF_DATASETS_OFFLINE=1

# 4. Rutas
DIR_PROYECTO="/gpfs/projects/ehpc485/tesi681824/transcriptor"
ORIGIN_DATOS="$DIR_PROYECTO/datos"
DEST="$TMPDIR"

echo "----------------------------------------------------------------"
echo "INICIO PREPARACION DE DATOS EN SSD LOCAL"
date
echo "----------------------------------------------------------------"

# 5. Copiar datos crudos al SSD
echo "Copiando carpeta 'datos' al disco SSD temporal..."
cp -r $ORIGIN_DATOS $DEST/
echo "Copia finalizada."

echo "----------------------------------------------------------------"
echo "ESTADO DEL DISCO SSD LOCAL:"
df -h $DEST
echo "----------------------------------------------------------------"

# 6. Variables de entorno para que el script de Python sepa donde leer y escribir
export LOCAL_DATOS="$DEST/datos"
export LOCAL_OUT="$DEST/dataset_unificado"

# 7. Ejecutar el script de preparacion
cd $DIR_PROYECTO
echo "Procesando audios a 16kHz y generando dataset..."
python -u preparar_datos_completo.py

# 8. RESCATE DEL DATASET (CRITICO)
echo "Copiando el dataset unificado de vuelta al almacenamiento persistente (GPFS)..."
cp -r $LOCAL_OUT $DIR_PROYECTO/

echo "----------------------------------------------------------------"
echo "FIN DEL TRABAJO. Dataset guardado a salvo en GPFS."
date
echo "----------------------------------------------------------------"