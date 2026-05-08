module Config

using TOML: parsefile

function load_imagenet_config(config_path="config.toml")
    @assert isfile(config_path) "File `$config_path` does not exist or script run from wrong path."
    cfg = parsefile(config_path)
    @assert haskey(cfg, "paths") "config file should have [paths] section"
    for key in ("path_to_db", "imagenet_path", "imagenet_preprocessed_path")
        @assert haskey(cfg["paths"], key) "config.toml missing paths.$key"
    end
    paths = cfg["paths"]
    return paths["path_to_db"], paths["imagenet_path"], paths["imagenet_preprocessed_path"]
end

export load_imagenet_config

end # module Config
