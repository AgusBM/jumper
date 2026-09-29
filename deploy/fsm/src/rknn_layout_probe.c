/* Report the C compiler's view of rknn_tensor_attr, so the Rust mirror in
 * rknn.rs can be checked against the vendored header. Compiles on any host --
 * it includes the header but links nothing, which is what makes this checkable
 * off the board. */
#include <stddef.h>
#include "rknn_api.h"

size_t mjrl_rknn_attr_size(void)            { return sizeof(rknn_tensor_attr); }
size_t mjrl_rknn_attr_n_elems_offset(void)  { return offsetof(rknn_tensor_attr, n_elems); }
size_t mjrl_rknn_input_size(void)           { return sizeof(rknn_input); }
size_t mjrl_rknn_output_size(void)          { return sizeof(rknn_output); }
