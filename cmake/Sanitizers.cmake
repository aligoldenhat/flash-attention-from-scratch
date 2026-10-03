# Host-side sanitizers (ASan + UBSan), for bugs in CPU code: the harness,
# buffer handling, etc. They do NOT check kernels; use compute-sanitizer:
#   compute-sanitizer --tool memcheck  ./your_program   # out-of-bounds, misaligned
#   compute-sanitizer --tool racecheck ./your_program   # shared-memory races
#   compute-sanitizer --tool initcheck ./your_program   # reading uninitialized memory
#
# ASan and the CUDA driver disagree about memory layout, so run with:
#   ASAN_OPTIONS=protect_shadow_gap=0 ./your_program

function(enable_sanitizers target enabled)
  if(NOT enabled)
    return()
  endif()

  # Separate -fsanitize flags: -Xcompiler splits on commas, so
  # "-fsanitize=address,undefined" would be broken in half.
  set(flags -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
  list(JOIN flags "," flags_csv)

  target_compile_options(
    ${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${flags}>
                      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flags_csv}>)
  target_link_options(${target} PRIVATE -fsanitize=address -fsanitize=undefined)
endfunction()
