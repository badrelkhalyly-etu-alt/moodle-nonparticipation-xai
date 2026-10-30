# Data (not distributed)

The raw files contain institutional student records and are not included in this repository.
To reproduce the analysis, place the following files in this folder:

| File | Content |
|---|---|
| `udk_moodle_all_grades.csv` | Moodle gradebook export: `id, timemodified, userid, courseid, finalgrade, itemtype` (20,317 records) |
| `logs_*.csv` (4 files) | Moodle event logs (French export) of the four student-designed courses used in the case study |

The event logs contain names and IP addresses. The pipeline never writes them to its output: the
courses are reported only as C1 to C4, and the account that exported the logs is excluded.
