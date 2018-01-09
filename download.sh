#!/bin/bash

HOME=~
PROJECT_DIR=$HOME/tesina
WORKSPACE_DIR=$PROJECT_DIR/workspace
SOURCE_DIR=$PROJECT_DIR/tesina
FILENAME_TEMP="index_temp.html"
DATASET_LIST_FILE=$1
LOGIN_URL="http://mrgdatashare.robots.ox.ac.uk/accounts/login/"
MODELS_DIR=$PROJECT_DIR/models
EXTRINSICS_DIR=$PROJECT_DIR/extrinsics
CROP_WIDTH=880
CROP_HEIGHT=660
SCALE_WIDTH=128
SCALE_HEIGHT=96
FILENAME_COOKIE_TEMP=cookies.txt
# TODO why isn't sessionid mandatory?
#COOKIE_HEADER="Cookie: _ga=GA1.3.147376171.1502830394; _gat=1; _gid=GA1.3.370211867.1505683919; sessionid=gcef24ow0h95wjezgrqrkudsas1hp5x1"

# TODO handle authentication errors
printf "Username: "
read USERNAME
stty -echo
printf "Password: "
read PASSWORD
stty echo
printf "\n"

output_dir=$WORKSPACE_DIR
processing_dataset=false

function download_file { # 1 = url, 2 = user, 3 = pass, 4 = filename_path
    	wget --save-cookies $FILENAME_COOKIE_TEMP --server-response -q -O - $1 > $FILENAME_TEMP
	csrf_middleware_token=$(sed -n "/csrfmiddlewaretoken/s/.*name='csrfmiddlewaretoken'\s\+value='\([^']\+\).*/\1/p" $FILENAME_TEMP)
	next_redirect=$(sed -n '/next/s/.*name="next"\s\+value="\([^"]\+\).*/\1/p' $FILENAME_TEMP)
	next_redirect_encoded=$(php -r "echo urlencode(\"$next_redirect\");")
	# TODO why isn't referer header mandatory?
	#referer_header="Referer: ${referer_encoded}"
	post_data="csrfmiddlewaretoken="$csrf_middleware_token"&username="$2"&password="$3"&next="$next_redirect_encoded
	wget --load-cookies $FILENAME_COOKIE_TEMP --post-data="${post_data}" "${LOGIN_URL}" -O $4
}

while read url_dataset; do
        filename=$(basename $url_dataset)
        dirname=$(dirname $url_dataset)
	directory_name="${filename%.*}"
	filename_path="${output_dir}/${filename}"
        download_file $url_dataset $USERNAME $PASSWORD $filename_path	
	output_dataset_directory="${output_dir}/${directory_name}"
	mkdir -p $output_dataset_directory
	tar xopf "${filename_path}" -C $output_dataset_directory
	tar_output=$?
	if [ "$tar_output" -eq 0 ]; then
		rm "${filename_path}"
	fi
	if [ "$processing_dataset" = true ] ; then
		#wait $processing_dataset_pid
		processing_dataset=false
	fi		
	IFS='_' read -ra FOLDERS <<< "$filename"
	dataset_image_directory="${output_dataset_directory}/${FOLDERS[0]}/${FOLDERS[1]}/${FOLDERS[2]}"
        vo_filename="${FOLDERS[0]}_vo.tar"
        vo_filename_path="${output_dir}/${vo_filename}"
        url_vo_file="${dirname}/$vo_filename"        
        download_file $url_vo_file $USERNAME $PASSWORD $vo_filename_path
        output_vo_directory="${output_dataset_directory}/vo"
        mkdir -p $output_vo_directory
        tar xopf $vo_filename_path -C $output_vo_directory
        tar_vo_output=$?
        if [ "$tar_vo_output" -eq 0 ]; then
		rm "${vo_filename_path}"
	fi
        poses_file="${output_vo_directory}/${FOLDERS[0]}/vo/vo.csv"
	python "${SOURCE_DIR}/adapt_images.py" "$dataset_image_directory" "$poses_file" --models_dir "${MODELS_DIR}" --crop "${CROP_WIDTH}" "${CROP_HEIGHT}" --scale "${SCALE_WIDTH}" "${SCALE_HEIGHT}" --output_dir "${output_dataset_directory}" # &	
	# TODO comment this line when running in background	
	if [ "$tar_output" -eq 0 ]; then
		rm -rf "${dataset_image_directory}"
	fi
	processing_dataset_pid=$!
	processing_dataset=true
	
	rm $FILENAME_TEMP
done <$DATASET_LIST_FILE
rm $FILENAME_COOKIE_TEMP

