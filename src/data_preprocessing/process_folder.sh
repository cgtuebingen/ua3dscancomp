#!/usr/bin/bash

input_folder=$1

# for folder in $(find ${input_folder} -mindepth 1 -type d)
# do

folder=${input_folder}

result_folder=/graphics/scratch2/datasets/objaverse1.0_processed/$(basename "${folder}")
mkdir -p "${result_folder}"
echo "processing ${folder}"
echo "outputting to ${result_folder}"

for file in $(find ${folder} -type f)
do
    output="${result_folder}/$(basename "${file}" | sed s/.glb//)_joined.glb"
    if [ -f "${output}" ]
    then
        echo "${output} already exists"
        continue
    fi
    echo "processing ${file} into ${output}..."
    sem -j+0 blender -b -P combine_meshesOfOnceScene_intoOneMesh.py "${file}" "${output}"
done
sem --wait
# done
