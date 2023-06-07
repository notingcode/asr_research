readonly DATASETS_PATH='./speech_data'
readonly DATASET1='한국인 대화 음성'
readonly DATASET2='자유대화 음성(일반남녀)'

readonly DATASET3='013.구음장애 음성인식 데이터'
readonly DATASET3_SUBDIR='01.데이터'
readonly DATASET3_LABEL='라벨링데이터'
readonly DATASET3_SRC='원천데이터'

readonly DATASET4='한국어 음성'
readonly DATASET4_SUBDIR1='전시문_통합_스크립트'
readonly DATASET4_SUBDIR2='평가용_데이터'
readonly DATASET4_SUBDIR3='한국어_음성_분야'

readonly TRAINING='Training'
readonly VALIDATION='Validation'
readonly GOOGLE_DRIVE_PATH='/content/drive/MyDrive/asr_research'

mkdir -p "$DATASETS_PATH/$DATASET1/$TRAINING"
mkdir -p "$DATASETS_PATH/$DATASET2/$TRAINING"
mkdir -p "$DATASETS_PATH/$DATASET3/$DATASET3_SUBDIR/1.$TRAINING/$DATASET3_LABEL"
mkdir -p "$DATASETS_PATH/$DATASET4/$DATASET4_SUBDIR1"

find "$GOOGLE_DRIVE_PATH/$DATASET1/$TRAINING/" -name '\[라벨\]*.tar.gz' -exec bash -c 'cp "$0" "$1" &' "{}" "$DATASETS_PATH/$DATASET1/$TRAINING" \;
find "$GOOGLE_DRIVE_PATH/$DATASET2/$TRAINING/" -name '\[라벨\]*.zip' -exec bash -c 'cp "$0" "$1" &' "{}" "$DATASETS_PATH/$DATASET2/$TRAINING" \;
find "$GOOGLE_DRIVE_PATH/$DATASET3/$DATASET3_SUBDIR/1.$TRAINING/$DATASET3_LABEL/" -name 'TL01*.zip' -exec bash -c 'cp "$0" "$1" &' "{}" "$DATASETS_PATH/$DATASET3/$DATASET3_SUBDIR/1.$TRAINING/$DATASET3_LABEL/" \;
cp "$GOOGLE_DRIVE_PATH/$DATASET4/$DATASET4_SUBDIR1/KsponSpeech_scripts.zip" "$DATASETS_PATH/$DATASET4/$DATASET4_SUBDIR1" &