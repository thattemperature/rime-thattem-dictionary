((nil
  . ((eglot-workspace-configuration
      . (:pylsp
         (:plugins
          (:pyflakes
           (:enabled t)

           :mccabe
           (:enabled t)

           :pycodestyle
           (:enabled
            t

            :maxLineLength
            70

            :indentSize
            4)

           :black
           (:enabled
            t

            :line_length
            70))

          :signature
          (:formatter
           "black"

           :include_docstring
           t

           :line_length
           70)))))))
