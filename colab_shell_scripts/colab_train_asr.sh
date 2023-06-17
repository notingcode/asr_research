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

readonly DATASET4='한국어 음성'
readonly DATASET4_SUBDIR1='전시문_통합_스크립트'
readonly DATASET4_SUBDIR2='평가용_데이터'
readonly DATASET4_SUBDIR3='한국어_음성_분야'

# mkdir -p "$DATASETS_PATH/$DATASET1/$TRAINING"
# mkdir -p "$DATASETS_PATH/$DATASET1/$VALIDATION"
# mkdir -p "$DATASETS_PATH/$DATASET2/$TRAINING"
# mkdir -p "$DATASETS_PATH/$DATASET2/$VALIDATION"
mkdir -p "$DATASETS_PATH/$DATASET3/$DATASET3_SUBDIR/1.$TRAINING/$DATASET3_LABEL"
mkdir -p "$DATASETS_PATH/$DATASET3/$DATASET3_SUBDIR/1.$TRAINING/$DATASET3_SRC"
mkdir -p "$DATASETS_PATH/$DATASET3/$DATASET3_SUBDIR/2.$VALIDATION/$DATASET3_LABEL"
mkdir -p "$DATASETS_PATH/$DATASET3/$DATASET3_SUBDIR/2.$VALIDATION/$DATASET3_SRC"
mkdir -p "$DATASETS_PATH/$DATASET4/$DATASET4_SUBDIR1"
mkdir -p "$DATASETS_PATH/$DATASET4/$DATASET4_SUBDIR2"
mkdir -p "$DATASETS_PATH/$DATASET4/$DATASET4_SUBDIR3"

# #Dataset 1 Label
# #Train
# cp "$GOOGLE_DRIVE_PATH/$DATASET1/$TRAINING/[라벨]3.일상안부_dialog_01.tar.gz" "$DATASETS_PATH/$DATASET1/$TRAINING" &

# #Validation
# cp "$GOOGLE_DRIVE_PATH/$DATASET1/$VALIDATION/[라벨]3.일상안부_dialog_01.tar.gz" "$DATASETS_PATH/$DATASET1/$VALIDATION" &

# #Dataset 1 Source
# #Train
# cp "$GOOGLE_DRIVE_PATH/$DATASET1/$TRAINING/[원천]3.일상안부_dialog_01.tar" "$DATASETS_PATH/$DATASET1/$TRAINING" &

# #Validation
# cp "$GOOGLE_DRIVE_PATH/$DATASET1/$VALIDATION/[원천]3.일상안부_dialog_01.tar.gz" "$DATASETS_PATH/$DATASET1/$VALIDATION" &
# cp "$GOOGLE_DRIVE_PATH/$DATASET1/$VALIDATION/[원천]4.생활_life_01.tar.gz" "$DATASETS_PATH/$DATASET1/$VALIDATION" &

# ###################################################################################################################################################

# #Dataset 2 Label
# #Train
# cp "$GOOGLE_DRIVE_PATH/$DATASET2/$TRAINING/[라벨]3.스튜디오.zip" "$DATASETS_PATH/$DATASET2/$TRAINING" &

# #Validation
# cp "$GOOGLE_DRIVE_PATH/$DATASET2/$VALIDATION/[라벨]3.스튜디오.zip" "$DATASETS_PATH/$DATASET2/$VALIDATION" &

# #Dataset 2 Source
# #Train
# cp "$GOOGLE_DRIVE_PATH/$DATASET2/$TRAINING/[원천]3.스튜디오_1.zip" "$DATASETS_PATH/$DATASET2/$TRAINING" &

# #Validation
# cp "$GOOGLE_DRIVE_PATH/$DATASET2/$VALIDATION/[원천]3.스튜디오.zip" "$DATASETS_PATH/$DATASET2/$VALIDATION" &

###################################################################################################################################################

#Dataset 3 Label
#Train
cp "$GOOGLE_DRIVE_PATH/$DATASET3/$DATASET3_SUBDIR/1.$TRAINING/$DATASET3_LABEL/TL01_뇌신경장애.zip" "$DATASETS_PATH/$DATASET3/$DATASET3_SUBDIR/1.$TRAINING/$DATASET3_LABEL" &

#Validation
cp "$GOOGLE_DRIVE_PATH/$DATASET3/$DATASET3_SUBDIR/2.$VALIDATION/$DATASET3_LABEL/VL01_뇌신경장애.zip" "$DATASETS_PATH/$DATASET3/$DATASET3_SUBDIR/2.$VALIDATION/$DATASET3_LABEL" &

#Dataset 3 Source
#Train
cp "$GOOGLE_DRIVE_PATH/$DATASET3/$DATASET3_SUBDIR/1.$TRAINING/$DATASET3_SRC/TS01_뇌신경장애.zip" "$DATASETS_PATH/$DATASET3/$DATASET3_SUBDIR/1.$TRAINING/$DATASET3_SRC" &

#Validation
cp "$GOOGLE_DRIVE_PATH/$DATASET3/$DATASET3_SUBDIR/2.$VALIDATION/$DATASET3_SRC/VS01_뇌신경장애.zip" "$DATASETS_PATH/$DATASET3/$DATASET3_SUBDIR/2.$VALIDATION/$DATASET3_SRC" &

#Dataset 4 Label
find "$GOOGLE_DRIVE_PATH/$DATASET4/$DATASET4_SUBDIR1" -name '*.zip' -exec bash -c 'cp "$0" "$1" &' "{}" "$DATASETS_PATH/$DATASET4/$DATASET4_SUBDIR1" \;

#Dataset 4 Source
find "$GOOGLE_DRIVE_PATH/$DATASET4/$DATASET4_SUBDIR2" -name '*.zip' -exec bash -c 'cp "$0" "$1" &' "{}" "$DATASETS_PATH/$DATASET4/$DATASET4_SUBDIR2" \;
find "$GOOGLE_DRIVE_PATH/$DATASET4/$DATASET4_SUBDIR3" -name '*.zip' -exec bash -c 'cp "$0" "$1" &' "{}" "$DATASETS_PATH/$DATASET4/$DATASET4_SUBDIR3" \;
