readonly DATASET_DIR_NAME='./speech_data'
readonly DATASET_NAME_1='한국인 대화 음성'
readonly DATASET_NAME_2='자유대화 음성(일반남녀)'
readonly TRAINING_STRING='Training'
readonly VALIDATION_STRING='Validation'
readonly GOOGLE_DRIVE_PATH='/content/drive/MyDrive/asr_research'

mkdir -p "$DATASET_DIR_NAME/$DATASET_NAME_1/$TRAINING_STRING"
mkdir -p "$DATASET_DIR_NAME/$DATASET_NAME_1/$VALIDATION_STRING"
mkdir -p "$DATASET_DIR_NAME/$DATASET_NAME_2/$TRAINING_STRING"
mkdir -p "$DATASET_DIR_NAME/$DATASET_NAME_2/$VALIDATION_STRING"

find "$GOOGLE_DRIVE_PATH/$DATASET_NAME_1/$TRAINING_STRING/" -name '\[라벨\]*.tar.gz' -exec bash -c 'cp "$0" "$1" &' "{}" "$DATASET_DIR_NAME/$DATASET_NAME_1/$TRAINING_STRING" \;
find "$GOOGLE_DRIVE_PATH/$DATASET_NAME_2/$TRAINING_STRING/" -name '\[라벨\]*.zip' -exec bash -c 'cp "$0" "$1" &' "{}" "$DATASET_DIR_NAME/$DATASET_NAME_2/$TRAINING_STRING" \;