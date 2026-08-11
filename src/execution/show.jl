function Base.show(io::IO, session::LocalRunSession)
    print(
        io,
        "LocalRunSession(",
        nameof(typeof(session.run)),
        "; status=",
        run_status(session),
        ", threaded=",
        session.threaded,
        ")",
    )
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", session::LocalRunSession) = show(io, session)
