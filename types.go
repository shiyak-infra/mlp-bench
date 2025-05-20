package main

const (
	LabelNodeBanchResults = "infra.shiyak.com/bench-results"
)

type Mode string

const (
	ModePrecheck = "precheck"
	ModeBench    = "bench"
	ModeDryrun   = "dryrun"
)

func (mode Mode) Valid() bool {
	if mode != ModePrecheck && mode != ModeBench && mode != ModeDryrun {
		return false
	}
	return true
}
