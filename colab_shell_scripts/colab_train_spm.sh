readonly DATASETS_PATH='./speech_data'
readonly DATASET1='한국인 대화 음성'
readonly DATASET2='자유대화 음성(일반남녀)'
readonly DATASET3='013.구음장애 음성인식 데이터'
readonly DATASET3_SUBDIR='01.데이터'
readonly DATASET3_LABEL='라벨링데이터'
readonly DATASET3_SRC='원천데이터'
readonly TRAINING='Training'
readonly VALIDATION='Validation'
readonly GOOGLE_DRIVE_PATH='/content/drive/MyDrive/asr_research'

mkdir -p "$DATASETS_PATH/$DATASET1/$TRAINING"
mkdir -p "$DATASETS_PATH/$DATASET1/$VALIDATION"
mkdir -p "$DATASETS_PATH/$DATASET2/$TRAINING"
mkdir -p "$DATASETS_PATH/$DATASET2/$VALIDATION"

find "$GOOGLE_DRIVE_PATH/$DATASET1/$TRAINING/" -name '\[라벨\]*.tar.gz' -exec bash -c 'cp "$0" "$1" &' "{}" "$DATASETS_PATH/$DATASET1/$TRAINING" \;
find "$GOOGLE_DRIVE_PATH/$DATASET2/$TRAINING/" -name '\[라벨\]*.zip' -exec bash -c 'cp "$0" "$1" &' "{}" "$DATASETS_PATH/$DATASET2/$TRAINING" \;