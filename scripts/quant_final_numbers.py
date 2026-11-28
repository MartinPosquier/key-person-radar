"""Owner: quant session. Task Q13: writes docs/final/numbers/quant.json.

Thin wrapper around kpr.validation.final_numbers (tested in tests/quant/). Run it in week 4
on the real tables, after task M12, with ``--include-test``.
"""

from kpr.validation.final_numbers import main

if __name__ == "__main__":
    raise SystemExit(main())
