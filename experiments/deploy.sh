# build docker only if --build is passed

build_docker=false

while [ "$1" != "" ]; do
    case $1 in
        --build )               build_docker=true
                                ;;
    esac
    shift
done

if [ "$build_docker" = true ]; then
    echo "Building docker image"
    cdMATS
    echo pwd: $(pwd)
    docker build . --target dev -t cybershiptrooper/db_fresh_circuits --build-arg github_token=${github_token}
fi

k delete jobs --all
cdMATS
cd circuits-benchmark/experiments
k create -f fresh_circuits_db.yaml