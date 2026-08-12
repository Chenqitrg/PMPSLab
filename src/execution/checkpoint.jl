const _CHECKPOINT_MAGIC = "PMPSLab checkpoint"
const _CHECKPOINT_SCHEMA_VERSION = UInt8(1)

struct CheckpointHeader
    magic::String
    schema_version::UInt8
    algorithm::Symbol
    package_version::VersionNumber
end

function _checkpoint_algorithm(snapshot::AbstractAlgorithmSnapshot)
    throw(ArgumentError(
        "unsupported checkpoint snapshot type $(typeof(snapshot))"
    ))
end

_checkpoint_algorithm(::SplitSnapshot) = :split
_checkpoint_algorithm(::LoopTNRSnapshot) = :looptnr

function _checkpoint_header(snapshot::AbstractAlgorithmSnapshot)
    return CheckpointHeader(
        _CHECKPOINT_MAGIC,
        _CHECKPOINT_SCHEMA_VERSION,
        _checkpoint_algorithm(snapshot),
        Base.pkgversion(@__MODULE__),
    )
end

function _write_checkpoint_file(
        path::AbstractString,
        header::CheckpointHeader,
        saved::AbstractAlgorithmSnapshot,
    )
    JLD2.jldopen(path, "w") do file
        file["header/magic"] = header.magic
        file["header/schema_version"] = header.schema_version
        file["header/algorithm"] = String(header.algorithm)
        file["header/package_version"] = string(header.package_version)
        file["snapshot"] = saved
    end
    return nothing
end

function _read_checkpoint_header(file)
    magic = file["header/magic"]
    magic == _CHECKPOINT_MAGIC || throw(ArgumentError(
        "invalid PMPSLab checkpoint magic value"
    ))

    schema_version = file["header/schema_version"]
    schema_version == _CHECKPOINT_SCHEMA_VERSION || throw(ArgumentError(
        "unsupported checkpoint schema version $schema_version"
    ))

    algorithm = Symbol(file["header/algorithm"])
    package_version = VersionNumber(file["header/package_version"])
    current_version = Base.pkgversion(@__MODULE__)
    package_version == current_version || throw(ArgumentError(
        "checkpoint was written by PMPSLab $package_version, but the loaded version is " *
        "$current_version"
    ))

    return CheckpointHeader(
        magic, schema_version, algorithm, package_version
    )
end

"""
    write_checkpoint(path::AbstractString, snapshot::AbstractAlgorithmSnapshot)

Write a versioned PMPSLab checkpoint as a JLD2 file. The file is first written
beside `path` and is published with a rename only after JLD2 closes it
successfully. The destination directory must already exist.

The checkpoint promises recovery only in the same PMPSLab and dependency
environment. A platform adapter may upload the completed file after this
function returns.
"""
function write_checkpoint(
        path::AbstractString,
        saved::AbstractAlgorithmSnapshot,
    )
    header = _checkpoint_header(saved)
    destination = abspath(path)
    directory = dirname(destination)
    isdir(directory) || throw(ArgumentError(
        "checkpoint directory does not exist: $directory"
    ))

    temporary_path, temporary_io = mktemp(directory; cleanup=false)
    close(temporary_io)
    try
        _write_checkpoint_file(temporary_path, header, saved)
        Base.Filesystem.rename(temporary_path, destination)
    catch
        ispath(temporary_path) && rm(temporary_path)
        rethrow()
    end
    return nothing
end

"""
    read_checkpoint(path::AbstractString) -> AbstractAlgorithmSnapshot

Read and validate a PMPSLab JLD2 checkpoint from `path`.
"""
function read_checkpoint(path::AbstractString)
    return JLD2.jldopen(path, "r") do file
        header = _read_checkpoint_header(file)
        saved = file["snapshot"]
        saved isa AbstractAlgorithmSnapshot || throw(ArgumentError(
            "checkpoint payload is not an algorithm snapshot"
        ))
        actual_algorithm = _checkpoint_algorithm(saved)
        actual_algorithm == header.algorithm || throw(ArgumentError(
            "checkpoint header declares $(header.algorithm), but payload contains " *
            "$actual_algorithm"
        ))
        return saved
    end
end
