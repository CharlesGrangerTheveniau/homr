# ruff: noqa: E501

import re
import unittest
from typing import Any

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from training.datasets.music_xml_parser import music_xml_string_to_tokens
from training.transformer.training_vocabulary import (
    read_token_lines,
    token_lines_to_str,
)


class TestDynamicsParser(unittest.TestCase):
    """Tests for parsing MusicXML dynamics and wedges into tokens."""

    def test_parse_dynamic_pp(self) -> None:
        example = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>1</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
      </attributes>
      <direction placement="below">
        <direction-type>
          <dynamics><pp/></dynamics>
        </direction-type>
        <staff>1</staff>
      </direction>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>16</duration>
        <voice>1</voice>
        <type>whole</type>
        <staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>"""
        tokens = music_xml_string_to_tokens(example)
        flat = [x for xxs in tokens for xs in xxs for x in xs]
        token_str = token_lines_to_str(flat)
        expected = """clef_G2 _ _ _ upper
keySignature_0 . . . .
timeSignature/4 . . . .
dynamic_pp _ _ _ upper
note_1 C4 _ _ upper
barline . . . ."""
        self.assertEqual(token_str, expected)

    def test_parse_dynamic_ff(self) -> None:
        example = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>1</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>8</duration>
        <voice>1</voice>
        <type>half</type>
        <staff>1</staff>
      </note>
      <direction placement="below">
        <direction-type>
          <dynamics><ff/></dynamics>
        </direction-type>
        <staff>1</staff>
      </direction>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>8</duration>
        <voice>1</voice>
        <type>half</type>
        <staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>"""
        tokens = music_xml_string_to_tokens(example)
        flat = [x for xxs in tokens for xs in xxs for x in xs]
        token_str = token_lines_to_str(flat)
        expected = """clef_G2 _ _ _ upper
keySignature_0 . . . .
timeSignature/4 . . . .
note_2 C4 _ _ upper
dynamic_ff _ _ _ upper
note_2 E4 _ _ upper
barline . . . ."""
        self.assertEqual(token_str, expected)

    def test_parse_crescendo_hairpin(self) -> None:
        example = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>1</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
      </attributes>
      <direction placement="below">
        <direction-type>
          <wedge type="crescendo"/>
        </direction-type>
        <staff>1</staff>
      </direction>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>8</duration>
        <voice>1</voice>
        <type>half</type>
        <staff>1</staff>
      </note>
      <direction placement="below">
        <direction-type>
          <wedge type="stop"/>
        </direction-type>
        <staff>1</staff>
      </direction>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>8</duration>
        <voice>1</voice>
        <type>half</type>
        <staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>"""
        tokens = music_xml_string_to_tokens(example)
        flat = [x for xxs in tokens for xs in xxs for x in xs]
        token_str = token_lines_to_str(flat)
        expected = """clef_G2 _ _ _ upper
keySignature_0 . . . .
timeSignature/4 . . . .
crescendoStart _ _ _ upper
note_2 C4 _ _ upper
crescendoEnd _ _ _ upper
note_2 E4 _ _ upper
barline . . . ."""
        self.assertEqual(token_str, expected)

    def test_parse_diminuendo_hairpin(self) -> None:
        example = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>1</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
      </attributes>
      <direction placement="below">
        <direction-type>
          <wedge type="diminuendo"/>
        </direction-type>
        <staff>1</staff>
      </direction>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>8</duration>
        <voice>1</voice>
        <type>half</type>
        <staff>1</staff>
      </note>
      <direction placement="below">
        <direction-type>
          <wedge type="stop"/>
        </direction-type>
        <staff>1</staff>
      </direction>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>8</duration>
        <voice>1</voice>
        <type>half</type>
        <staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>"""
        tokens = music_xml_string_to_tokens(example)
        flat = [x for xxs in tokens for xs in xxs for x in xs]
        token_str = token_lines_to_str(flat)
        expected = """clef_G2 _ _ _ upper
keySignature_0 . . . .
timeSignature/4 . . . .
diminuendoStart _ _ _ upper
note_2 C4 _ _ upper
diminuendoEnd _ _ _ upper
note_2 E4 _ _ upper
barline . . . ."""
        self.assertEqual(token_str, expected)

    def test_parse_dynamics_grand_staff(self) -> None:
        """Dynamics on staff 2 should get lower position."""
        example = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>16</duration>
        <voice>1</voice>
        <type>whole</type>
        <staff>1</staff>
      </note>
      <backup><duration>16</duration></backup>
      <direction placement="below">
        <direction-type>
          <dynamics><mf/></dynamics>
        </direction-type>
        <staff>2</staff>
      </direction>
      <note>
        <pitch><step>C</step><octave>3</octave></pitch>
        <duration>16</duration>
        <voice>5</voice>
        <type>whole</type>
        <staff>2</staff>
      </note>
    </measure>
  </part>
</score-partwise>"""
        tokens = music_xml_string_to_tokens(example)
        flat = [x for xxs in tokens for xs in xxs for x in xs]
        token_str = token_lines_to_str(flat)
        expected = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_0 . . . .
timeSignature/4 . . . .
note_1 C4 _ _ upper
dynamic_mf _ _ _ lower
note_1 C3 _ _ lower
barline . . . ."""
        self.assertEqual(token_str, expected)

    def test_parse_multiple_dynamics_in_direction(self) -> None:
        """A direction with multiple direction-type children."""
        example = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>1</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
      </attributes>
      <direction placement="below">
        <direction-type>
          <dynamics><sfz/></dynamics>
        </direction-type>
        <staff>1</staff>
      </direction>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>16</duration>
        <voice>1</voice>
        <type>whole</type>
        <staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>"""
        tokens = music_xml_string_to_tokens(example)
        flat = [x for xxs in tokens for xs in xxs for x in xs]
        token_str = token_lines_to_str(flat)
        self.assertIn("dynamic_sfz _ _ _ upper", token_str)


class TestDynamicsGenerator(unittest.TestCase):
    """Tests for generating MusicXML from dynamics tokens."""

    def test_generate_dynamic_pp(self) -> None:
        token_str = """clef_G2 _ _ _ upper
keySignature_0 . . . .
timeSignature/4 . . . .
dynamic_pp _ _ _ upper
note_1 C4 _ _ upper
barline . . . ."""
        tokens = read_token_lines(token_str.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        xml_str = self._xml_to_str(xml)
        self.assertIn("XMLDirection", xml_str)
        self.assertIn("XMLDynamics", xml_str)
        self.assertIn("XMLPp", xml_str)

    def test_generate_crescendo_wedge(self) -> None:
        token_str = """clef_G2 _ _ _ upper
keySignature_0 . . . .
timeSignature/4 . . . .
crescendoStart _ _ _ upper
note_2 C4 _ _ upper
crescendoEnd _ _ _ upper
note_2 E4 _ _ upper
barline . . . ."""
        tokens = read_token_lines(token_str.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        xml_str = self._xml_to_str(xml)
        self.assertIn("XMLWedge", xml_str)
        # Should have two wedge elements (start and stop)
        self.assertEqual(xml_str.count("XMLWedge"), 2)

    def test_generate_diminuendo_wedge(self) -> None:
        token_str = """clef_G2 _ _ _ upper
keySignature_0 . . . .
timeSignature/4 . . . .
diminuendoStart _ _ _ upper
note_2 C4 _ _ upper
diminuendoEnd _ _ _ upper
note_2 E4 _ _ upper
barline . . . ."""
        tokens = read_token_lines(token_str.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        xml_str = self._xml_to_str(xml)
        self.assertIn("XMLWedge", xml_str)
        self.assertEqual(xml_str.count("XMLWedge"), 2)

    def test_generate_dynamic_with_staff(self) -> None:
        token_str = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_0 . . . .
timeSignature/4 . . . .
dynamic_f _ _ _ lower
note_1 C4 _ _ upper&note_1 C3 _ _ lower
barline . . . ."""
        tokens = read_token_lines(token_str.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        xml_str = self._xml_to_str(xml)
        self.assertIn("XMLDirection", xml_str)
        self.assertIn("XMLF()", xml_str)
        self.assertIn("XMLStaff(value: 2)", xml_str)

    def test_roundtrip_dynamics(self) -> None:
        """Parse MusicXML with dynamics, generate back, verify dynamics present."""
        source = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>1</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
      </attributes>
      <direction placement="below">
        <direction-type>
          <dynamics><p/></dynamics>
        </direction-type>
        <staff>1</staff>
      </direction>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>16</duration>
        <voice>1</voice>
        <type>whole</type>
        <staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>"""
        tokens = music_xml_string_to_tokens(source)
        flat = [x for xxs in tokens for xs in xxs for x in xs]

        xml = generate_xml(XmlGeneratorArguments(), [flat], "")
        xml_str = self._xml_to_str(xml)
        self.assertIn("XMLDynamics", xml_str)
        self.assertIn("XMLP()", xml_str)

    def _xml_to_str(self, xml: Any) -> str:
        def recurse(node_or_list: Any) -> str:
            if isinstance(node_or_list, list):
                return (
                    "["
                    + ",".join(recurse(child) for child in node_or_list if child is not None)
                    + "]"
                )

            node = node_or_list
            name = node.__class__.__name__

            ignore_nodes = (
                "XMLAlter",
                "XMLOctave",
                "XMLType",
                "XMLPartList",
                "XMLDefaults",
            )

            if name in ignore_nodes:
                return ""
            value = getattr(node, "value_", None)

            if hasattr(node, "children"):
                children = node.children
            elif hasattr(node, "get_children"):
                children = node.get_children()
            else:
                children = []

            child_strs = [recurse(child) for child in children if child is not None]
            child_strs = [child for child in child_strs if child != ""]

            parts = []
            if value is not None and value != "":
                parts.append(f"value: {value}")
            if child_strs:
                parts.append(f"[{','.join(child_strs)}]")

            if parts:
                return f"{name}({','.join(parts)})"
            else:
                return f"{name}()"

        return recurse(xml)
