import math
import types

import torch
import torch.nn as nn


def fix_dinov2_for_onnx_export(model_dinov2):
    """This function fixes the Vision Transformer for ONNX export

    The Vision Transformers as defined in the DINOv2 can't be directly exported because
    of `upsample_bicubic2d_aa` is not a supported ONNX operation.
    So we have to replace it with a bilinear upsampling before export.

    :param model_dinov2: The DINOv2 model from Torch Hub

    :return: The model with interpolate_pos_encoding fixed for export
    """

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension

        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)

        # this is the part that is fixed
        aa = patch_pos_embed.reshape(
            1, int(math.sqrt(N)), int(math.sqrt(N)), dim
        ).permute(0, 3, 1, 2)
        bb = (w0 / math.sqrt(N), h0 / math.sqrt(N))
        cc = bb

        if True and isinstance(bb[0], torch.Tensor):
            cc = (bb[0].item(), bb[1].item())

        patch_pos_embed = nn.functional.interpolate(
            aa, scale_factor=cc, mode="bilinear"
        )

        print(w0, h0, patch_pos_embed.shape[-2:])
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    model_dinov2.backbone.interpolate_pos_encoding = types.MethodType(
        interpolate_pos_encoding, model_dinov2.backbone
    )

    return model_dinov2


def export_to_onnx(
    model,
    export_name="model.onnx",
    input_size=(1, 3, 518, 518),  # default DINOv2 train size
    use_dynamo=True,
):
    """A wrapper function to export a torch model to ONNX

    :param model: The model to export
    :param export_name: Name of the file to export to
    :param input_size: The dummy inut size used for the graph calculation
    :param use_dynamo: A flasg to set the use of the new export API

    :return: An ONNX program
    """

    model = fix_dinov2_for_onnx_export(model)

    x = torch.randn(*input_size)

    onnx_program = torch.onnx.export(
        model,
        x,
        export_name,
        export_params=True,
        input_names=["input"],
        dynamo=use_dynamo,
        verify=True,
        profile=True,
        dump_exported_program=True,
        artifacts_dir="outputs",
    )

    return onnx_program
