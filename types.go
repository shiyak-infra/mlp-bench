package main

const (
	LabelNodeBanchResults = "infra.shiyak.com/bench-results"
)

type Mode string

const (
	ModePrecheck = "precheck"
	ModeBench    = "bench"
)

func (mode Mode) Valid() bool {
	if mode != ModePrecheck && mode != ModeBench {
		return false
	}
	return true
}
