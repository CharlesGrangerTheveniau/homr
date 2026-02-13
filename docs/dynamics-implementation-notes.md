# Dynamics Implementation Notes

## 1. Dynamics Count Script

**File**: `scripts/count_dynamics_in_lieder.py`

Run with:
```bash
python scripts/count_dynamics_in_lieder.py
```

Scans all `.musicxml` files under `datasets/Lieder-main/` (both `flat/` and `scores/`).
Reports: total files with dynamics, event counts by type, hairpin counts, and coverage
analysis against the proposed vocabulary tokens.

Note: Requires the Lieder dataset to be downloaded and converted first via
`convert_lieder.py`.


## 2. XMLDirection Parser (Training Side)

**File**: `training/datasets/music_xml_parser.py`

### Changes
- Added `DYNAMICS_TOKENS` constant with all supported dynamic marking names
- Added `_process_direction()` function that handles `<direction>` MusicXML elements
- Wired `_process_direction()` into `_music_part_to_tokens()` main loop

### How it works
- Extracts `<dynamics>` children from `<direction-type>` elements. The dynamic name
  is the tag name of the child element (e.g., `<pp/>` -> "pp")
- Extracts `<wedge>` elements for crescendo/diminuendo hairpins
- Tracks wedge state per staff to pair stops with the correct start type
- Uses `append_symbol_to_staff()` to preserve grand staff assignment
- Dynamics tokens use `empty` ("_") for pitch, lift, and articulation branches

### Token format
```
dynamic_pp _ _ _ upper    # pp dynamic on upper staff
crescendoStart _ _ _ lower  # hairpin start on lower staff
diminuendoEnd _ _ _ upper   # hairpin stop on upper staff
```

### Tests
- `tests/test_dynamics.py::TestDynamicsParser` - 6 test cases covering:
  - Single dynamic (pp, ff)
  - Crescendo hairpin (start + stop)
  - Diminuendo hairpin (start + stop)
  - Grand staff dynamics (staff 2 -> lower position)
  - Multiple dynamics types (sfz)
- Updated `tests/test_music_xml_parser.py` existing tests to include dynamics output


## 3. MusicXML Dynamics Output (Generation Side)

**File**: `homr/music_xml_generator.py`

### Changes
- Added `DYNAMIC_XML_CLASSES` mapping from token names to musicxml library classes
- Added `build_dynamic_direction()` - creates MusicXML `<direction>` with `<dynamics>`
- Added `build_wedge_direction()` - creates MusicXML `<direction>` with `<wedge>`
- Added cases in `build_measures()` for `dynamic_*` and `crescendo*/diminuendo*` tokens

### Output format
```xml
<!-- dynamic_pp token produces: -->
<direction placement="below">
  <direction-type>
    <dynamics><pp/></dynamics>
  </direction-type>
  <staff>1</staff>
</direction>

<!-- crescendoStart token produces: -->
<direction placement="below">
  <direction-type>
    <wedge type="crescendo"/>
  </direction-type>
  <staff>1</staff>
</direction>

<!-- crescendoEnd/diminuendoEnd tokens produce: -->
<direction placement="below">
  <direction-type>
    <wedge type="stop"/>
  </direction-type>
  <staff>1</staff>
</direction>
```

### Tests
- `tests/test_dynamics.py::TestDynamicsGenerator` - 5 test cases covering:
  - Dynamic pp generation
  - Crescendo wedge generation
  - Diminuendo wedge generation
  - Dynamics with staff assignment (lower -> staff 2)
  - Round-trip: parse MusicXML -> tokens -> generate MusicXML

### Supporting changes
- `homr/transformer/vocabulary.py`: Updated `has_rhythm_symbol_a_position()` to include
  `dynamic`, `crescendo`, `diminuendo` prefixes (needed for position assignment)
- `training/datasets/staff_merging.py`: Updated `create_chord_over_two_staffs()` to
  handle dynamics as a separate category (standalone, not grouped into chords)


## 4. Vocabulary Expansion Blast Radius

**Uncommenting tokens in `vocabulary.py:72-83` adds 14 new rhythm tokens:**
- `dynamic_ppp`, `dynamic_pp`, `dynamic_p`, `dynamic_mp`, `dynamic_mf`
- `dynamic_f`, `dynamic_ff`, `dynamic_fff`, `dynamic_sfz`, `dynamic_fp`
- `crescendoStart`, `crescendoEnd`, `diminuendoStart`, `diminuendoEnd`

This changes `len(vocab.rhythm)` from 104 to 118.

### What breaks

| Component | File | Impact |
|-----------|------|--------|
| Decoder output layers | `training/architecture/transformer/decoder.py:62-66` | `nn.Linear(dim, config.num_rhythm_tokens)` creates mismatched shapes vs checkpoint weights |
| Token embeddings | `training/architecture/transformer/decoder.py:45` | `TokenEmbedding(config.decoder_dim, config.num_rhythm_tokens)` mismatches saved weights |
| Note mask tensor | `training/architecture/transformer/decoder.py:253` | `torch.zeros(config.num_rhythm_tokens)` has wrong size |
| Config propagation | `homr/transformer/configs.py:92-96` | `num_rhythm_tokens = len(vocab.rhythm)` cascades new size everywhere |
| ONNX conversion | `training/onnx/convert.py:128-145` | Loads old checkpoints into model with new vocab size -> shape mismatch |
| Inference decoder | `homr/transformer/decoder_inference.py:30-34` | Inverse vocab mapping mismatches ONNX model output indices |
| All existing weights | `homr/transformer/*.onnx`, `*.pth` | Trained with 104 rhythm tokens, incompatible with 118 |

### Required actions to enable dynamics
1. Uncomment the vocabulary tokens
2. **Retrain the model** with the new vocabulary (weights must match new dimensions)
3. Re-export ONNX models from new checkpoint
4. Update tokenizer JSON files if needed
5. Run full validation suite

### What does NOT break
- `staff2score.py` hardcoded indices (BOS=1, PAD=0) are at vocab start, unaffected
- `training_vocabulary.py` sort key uses `len(vocab.rhythm)` but only affects ordering, not correctness
- Parser code already emits dynamics tokens regardless of vocabulary membership
- Generator code handles dynamics tokens when present, falls through to `eprint()` when not


## 5. Paired Symbol Handling Pattern

### How the codebase handles existing paired symbols

**Slurs (slurStart / slurStop)**
- Stored in the **articulation branch** of `EncodedSymbol`, not as standalone rhythm tokens
- Extracted from `<slur type="start|stop">` in `_collect_articulation()` (parser)
- Written back via `build_articulations()` → `XMLSlur(type="start|stop")` (generator)
- **No explicit pairing validation** - start/stop are emitted independently
- `strip_slur_ties()` in `SymbolChord` can remove them for processing purposes
- Currently disabled in generator: slur/tie output is commented out (`# Disabled slurs and ties until the detection is more robust`)

**Ties (tieStart / tieStop)**
- Same pattern as slurs: stored in articulation branch
- No pairing validation

**Repeats (repeatStart / repeatEnd / repeatEndStart)**
- Standalone rhythm tokens (like dynamics will be)
- `_cleanup_barlines_and_repeats()` merges adjacent `repeatStart` + `repeatEnd` into `repeatEndStart`
- No explicit pairing validation beyond this merge

**Volta endings (voltaStart / voltaStop / voltaDiscontinue)**
- Standalone rhythm tokens
- `ConversionState` tracks volta numbers to associate start/stop with correct ending number
- No validation that every start has a matching stop

### Recommended approach for hairpins
Based on the existing patterns, hairpins should:
1. Be emitted as **standalone rhythm tokens** (already implemented)
2. **No explicit pairing validation** during parsing or generation
3. If malformed sequences occur (start without stop), the generator outputs what it receives
4. Optional future enhancement: add post-processing similar to `_cleanup_barlines_and_repeats()`
   to clean up orphaned hairpins (e.g., remove unmatched starts at measure end)
5. The wedge state tracking in `_process_direction()` handles stop-to-start pairing during
   parsing (determines whether a `<wedge type="stop">` is a crescendo or diminuendo end)
