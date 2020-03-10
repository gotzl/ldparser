Small tool to parse MoTec ld files.

Primarily, this parser was written to parse telemetry information of ld files written by Assetto Corsa Competizione, see also the related project ['acctelemetry'](https://github.com/gotzl/acctelemetry). However, the parser should work with other ld files as well.
It was tested with the sample ld files that come with a MoTec Mi2 Pro installation as well as with files written by ACC.

The decoding of the ld file is solely based on reverse engineering the binary data.
As of now, ldx files are not considered.

## Usage
See the __main__ function on how to use the tool.

As an example, the __main__ function reads all ld files in a given directory and creates some plots. 
Invoke with

```bash
python ldparser.py /path/to/some/dir
```
